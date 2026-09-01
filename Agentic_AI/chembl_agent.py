"""Production-oriented LangGraph service for evidence-grounded drug intelligence.

The service keeps deterministic data retrieval separate from LLM synthesis. It never
sends an API key to ChEMBL/PubChem and never logs the key. This is educational code;
its output is not a clinical or regulatory decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import quote

import requests
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("chembl_agent")
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data"
HERG_TARGET = "CHEMBL240"


class UpstreamError(RuntimeError):
    """A recoverable upstream-data failure with a safe public message."""


class CompoundStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    cid: int
    smiles: str
    molecular_weight: float | None = None
    molecular_formula: str | None = None


class ChemblCompound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chembl_id: str
    preferred_name: str | None = None
    max_phase: float | None = None
    alogp: float | None = None


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    standard_type: str | None = None
    standard_value: float | None = None
    standard_units: str | None = None
    pchembl_value: float
    target_chembl_id: str | None = None
    target_name: str | None = None
    assay_chembl_id: str | None = None
    document_chembl_id: str | None = None


class ActivitySummary(BaseModel):
    count: int = 0
    target_count: int = 0
    median_pchembl: float | None = None
    median_values_by_type: dict[str, float] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    """Strict contract returned by the OpenAI structured-output call."""

    compound: str
    overall_risk: Literal["low", "moderate", "high", "insufficient_evidence"]
    confidence: Literal["low", "medium", "high"]
    executive_summary: str
    key_findings: list[str] = Field(min_length=1, max_length=8)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=8)
    recommended_next_steps: list[str] = Field(min_length=1, max_length=8)
    disclaimer: str


class AgentState(TypedDict, total=False):
    request_id: str
    compound: str
    structure: dict[str, Any]
    chembl: dict[str, Any]
    activities: list[dict[str, Any]]
    herg_activities: list[dict[str, Any]]
    evidence: dict[str, Any]
    assessment: dict[str, Any]
    errors: Annotated[list[str], lambda left, right: left + right]
    audit: Annotated[list[dict[str, Any]], lambda left, right: left + right]


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    request_timeout_seconds: float = 20.0
    max_activities: int = 100
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 3
    store_openai_response: bool = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise structured logs without credentials or raw prompts."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class EvidenceClient:
    """Validated, retried, bounded client for PubChem and ChEMBL."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
        self.session.headers.update({"User-Agent": "chembl-agent-tutorial/2.0"})

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.session.get(url, params=params, timeout=self.settings.request_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise UpstreamError(f"Upstream request failed for {url.split('/')[2]}") from exc
        if not isinstance(payload, dict):
            raise UpstreamError("Upstream returned an unexpected payload")
        return payload

    @lru_cache(maxsize=256)
    def pubchem_structure(self, compound: str) -> CompoundStructure:
        name = compound.strip()
        if not name or len(name) > 200:
            raise ValueError("compound must contain 1–200 characters")
        payload = self._get_json(
            f"{PUBCHEM_URL}/compound/name/{quote(name, safe='')}/property/SMILES,MolecularWeight,MolecularFormula/JSON"
        )
        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not rows:
            raise UpstreamError(f"PubChem has no structure for {name!r}")
        row = rows[0]
        smiles = row.get("SMILES") or row.get("ConnectivitySMILES") or row.get("CanonicalSMILES")
        if not smiles:
            raise UpstreamError("PubChem result did not include a SMILES structure")
        return CompoundStructure(
            name=name,
            cid=row["CID"],
            smiles=smiles,
            molecular_weight=_safe_float(row.get("MolecularWeight")),
            molecular_formula=row.get("MolecularFormula"),
        )

    @lru_cache(maxsize=256)
    def chembl_compound(self, compound: str) -> ChemblCompound:
        payload = self._get_json(
            f"{CHEMBL_URL}/molecule/search.json",
            params={"q": compound.strip(), "limit": 5},
        )
        molecules = payload.get("molecules") or []
        if not molecules:
            raise UpstreamError(f"ChEMBL has no molecule match for {compound!r}")
        exact = [m for m in molecules if (m.get("pref_name") or "").casefold() == compound.strip().casefold()]
        row = (exact or molecules)[0]
        chembl_id = (row.get("molecule_hierarchy") or {}).get("parent_chembl_id") or row.get("molecule_chembl_id")
        if not chembl_id:
            raise UpstreamError("ChEMBL result did not include a molecule identifier")
        props = row.get("molecule_properties") or {}
        return ChemblCompound(
            chembl_id=chembl_id,
            preferred_name=row.get("pref_name"),
            max_phase=_safe_float(row.get("max_phase")),
            alogp=_safe_float(props.get("alogp")),
        )

    @lru_cache(maxsize=512)
    def activities(self, chembl_id: str, target_id: str | None = None) -> tuple[Activity, ...]:
        params: dict[str, Any] = {
            "molecule_chembl_id": chembl_id,
            "pchembl_value__isnull": "false",
            "limit": min(self.settings.max_activities, 1000),
        }
        if target_id:
            params["target_chembl_id"] = target_id
        payload = self._get_json(f"{CHEMBL_URL}/activity.json", params=params)
        parsed: list[Activity] = []
        for row in payload.get("activities") or []:
            pchembl = _safe_float(row.get("pchembl_value"))
            if pchembl is None:
                continue
            parsed.append(Activity(
                standard_type=row.get("standard_type"),
                standard_value=_safe_float(row.get("standard_value")),
                standard_units=row.get("standard_units"),
                pchembl_value=pchembl,
                target_chembl_id=row.get("target_chembl_id"),
                target_name=row.get("target_pref_name"),
                assay_chembl_id=row.get("assay_chembl_id"),
                document_chembl_id=row.get("document_chembl_id"),
            ))
        return tuple(parsed)


def summarize_activities(rows: list[dict[str, Any]]) -> ActivitySummary:
    activities = [Activity.model_validate(row) for row in rows]
    grouped: dict[str, list[float]] = {}
    for activity in activities:
        if activity.standard_type and activity.standard_value is not None:
            grouped.setdefault(activity.standard_type, []).append(activity.standard_value)
    return ActivitySummary(
        count=len(activities),
        target_count=len({a.target_chembl_id for a in activities if a.target_chembl_id}),
        median_pchembl=round(statistics.median(a.pchembl_value for a in activities), 2) if activities else None,
        median_values_by_type={key: round(statistics.median(values), 3) for key, values in grouped.items()},
    )


def deterministic_fallback(compound: str, evidence: dict[str, Any], errors: list[str]) -> RiskAssessment:
    herg = ActivitySummary.model_validate(evidence.get("herg_summary", {}))
    if errors or not evidence:
        risk, confidence = "insufficient_evidence", "low"
    elif herg.count and (herg.median_pchembl or 0) >= 6:
        risk, confidence = "high", "medium"
    elif herg.count:
        risk, confidence = "moderate", "medium"
    else:
        risk, confidence = "insufficient_evidence", "low"
    return RiskAssessment(
        compound=compound,
        overall_risk=risk,
        confidence=confidence,
        executive_summary="Deterministic fallback used; an OpenAI-generated interpretation was not available.",
        key_findings=[f"Quantitative hERG records retrieved: {herg.count}"],
        evidence_gaps=errors or ["No quantitative hERG evidence was retrieved."],
        recommended_next_steps=["Review source assays and confirm findings with validated experimental methods."],
        disclaimer="Research-use decision support only; not medical, clinical, or regulatory advice.",
    )


class DrugIntelligenceService:
    def __init__(self, settings: Settings | None = None, evidence_client: EvidenceClient | None = None):
        self.settings = settings or Settings()
        self.evidence_client = evidence_client or EvidenceClient(self.settings)
        self._graph = self._build_graph()

    @staticmethod
    def _audit(node: str, started: float, status: str = "ok") -> list[dict[str, Any]]:
        return [{"node": node, "status": status, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}]

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            structure_future = pool.submit(self.evidence_client.pubchem_structure, state["compound"])
            chembl_future = pool.submit(self.evidence_client.chembl_compound, state["compound"])
            structure = chembl = None
            try:
                structure = structure_future.result()
            except (UpstreamError, ValueError) as exc:
                errors.append(str(exc))
            try:
                chembl = chembl_future.result()
            except (UpstreamError, ValueError) as exc:
                errors.append(str(exc))
        return {
            "structure": structure.model_dump() if structure else {},
            "chembl": chembl.model_dump() if chembl else {},
            "errors": errors,
            "audit": self._audit("retrieve_identity", started, "partial" if errors else "ok"),
        }

    def _bioactivity(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        chembl_id = state.get("chembl", {}).get("chembl_id")
        if not chembl_id:
            return {"errors": ["Bioactivity skipped because no ChEMBL ID was resolved."], "audit": self._audit("retrieve_bioactivity", started, "skipped")}
        errors: list[str] = []
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                all_future = pool.submit(self.evidence_client.activities, chembl_id, None)
                herg_future = pool.submit(self.evidence_client.activities, chembl_id, HERG_TARGET)
                activities = [row.model_dump() for row in all_future.result()]
                herg = [row.model_dump() for row in herg_future.result()]
        except UpstreamError as exc:
            activities, herg = [], []
            errors.append(str(exc))
        return {
            "activities": activities,
            "herg_activities": herg,
            "errors": errors,
            "audit": self._audit("retrieve_bioactivity", started, "partial" if errors else "ok"),
        }

    def _compile(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        evidence = {
            "compound": state["compound"],
            "structure": state.get("structure", {}),
            "chembl": state.get("chembl", {}),
            "activity_summary": summarize_activities(state.get("activities", [])).model_dump(),
            "herg_summary": summarize_activities(state.get("herg_activities", [])).model_dump(),
            "source_record_ids": sorted({
                row["document_chembl_id"] for row in state.get("activities", [])
                if row.get("document_chembl_id")
            })[:25],
        }
        return {"evidence": evidence, "audit": self._audit("compile_evidence", started)}

    def _assess(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            assessment = deterministic_fallback(state["compound"], state["evidence"], state.get("errors", []))
            return {"assessment": assessment.model_dump(), "errors": ["OPENAI_API_KEY is not configured; deterministic fallback used."], "audit": self._audit("assess", started, "fallback")}
        try:
            client = OpenAI(api_key=key, timeout=self.settings.openai_timeout_seconds, max_retries=self.settings.openai_max_retries)
            response = client.responses.parse(
                model=self.settings.model,
                instructions=(
                    "You are a pharmaceutical safety evidence synthesizer. Use only the supplied JSON evidence. "
                    "Do not infer absence of risk from absence of data. Separate measured evidence from uncertainty. "
                    "Do not provide dosing or patient-specific advice."
                ),
                input=json.dumps(state["evidence"], sort_keys=True),
                text_format=RiskAssessment,
                store=self.settings.store_openai_response,
                max_output_tokens=1200,
                metadata={"request_id": state["request_id"], "workflow": "chembl-agent-v2"},
                safety_identifier=hashlib.sha256(state["request_id"].encode()).hexdigest()[:32],
            )
            assessment = response.output_parsed
            if assessment is None:
                raise RuntimeError("OpenAI returned no parsed assessment")
            return {"assessment": assessment.model_dump(), "audit": self._audit("assess", started)}
        except Exception as exc:  # SDK errors are deliberately converted to a safe fallback.
            LOGGER.exception("OpenAI assessment failed request_id=%s type=%s", state["request_id"], type(exc).__name__)
            assessment = deterministic_fallback(state["compound"], state["evidence"], state.get("errors", []))
            return {"assessment": assessment.model_dump(), "errors": [f"OpenAI assessment failed ({type(exc).__name__}); deterministic fallback used."], "audit": self._audit("assess", started, "fallback")}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_identity", self._retrieve)
        graph.add_node("retrieve_bioactivity", self._bioactivity)
        graph.add_node("compile_evidence", self._compile)
        graph.add_node("assess", self._assess)
        graph.set_entry_point("retrieve_identity")
        graph.add_edge("retrieve_identity", "retrieve_bioactivity")
        graph.add_edge("retrieve_bioactivity", "compile_evidence")
        graph.add_edge("compile_evidence", "assess")
        graph.add_edge("assess", END)
        return graph.compile()

    def assess(self, compound: str, *, request_id: str | None = None) -> AgentState:
        clean_name = compound.strip()
        if not clean_name or len(clean_name) > 200:
            raise ValueError("compound must contain 1–200 characters")
        rid = request_id or str(uuid.uuid4())
        LOGGER.info("workflow_started request_id=%s compound=%s", rid, clean_name)
        result = self._graph.invoke({"request_id": rid, "compound": clean_name, "errors": [], "audit": []})
        LOGGER.info("workflow_completed request_id=%s errors=%d", rid, len(result.get("errors", [])))
        return result


__all__ = [
    "ActivitySummary", "DrugIntelligenceService", "EvidenceClient", "RiskAssessment",
    "Settings", "configure_logging", "deterministic_fallback", "summarize_activities",
]
