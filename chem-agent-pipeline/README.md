# ChemAgent Pro — Production Agentic Cheminformatics Pipeline

**Author:** Himanshu Goel | [himanshugoel.github.io](https://himanshugoel.github.io)

Inspired by: ChatInvent (AstraZeneca, 2026) · ChemCrow (Nat Mach Intell) · ChemGraph (Comms Chem 2026) · CACTUS · DrugPilot

---

## What This Is

A **complete end-to-end production pipeline** that evolves a simple agentic cheminformatics script
into a full industry-standard system with typed state, conditional routing, human-in-the-loop
checkpointing, REST API, audit trail, and structured reports.

---

## Architecture (10 Stages)

```
Stage 0  │ Project scaffolding, architecture design, constants
Stage 1  │ Tool layer — 12 pure cheminformatics functions (RDKit + ML)
Stage 2  │ Tool registry — schemas, centralized dispatch, error handling
Stage 3  │ Typed state (CompoundState TypedDict — LangGraph pattern)
Stage 4  │ Agent nodes — parse → descriptors → alerts → ADMET → toxicity
Stage 5  │ Conditional routing (route_after_risk / route_after_hitl)
Stage 6  │ Human-in-the-loop checkpoint (HIGH risk gate)
Stage 7  │ Session checkpointing (SQLite → Postgres in production)
Stage 8  │ Report generator (JSON + Markdown + audit trail)
Stage 9  │ FastAPI REST API (POST /analyze, /batch, /hitl, /session)
Stage 10 │ CLI, unit tests, demo runner
```

---

## Pipeline Graph

```
Input SMILES
     │
     ▼
[parse_input] ──fail──► [report]
     │
     ▼
[compute_descriptors]   ← Ro5, LogP, TPSA, QED, CNS MPO
     │
     ▼
[screen_alerts]         ← PAINS, ICH M7 (Class 1/2/3), toxicophores
     │
     ▼
[predict_admet]         ← BBB, Caco-2, CYP3A4, hERG rapid, t½
     │
     ▼
[predict_toxicity]      ← hERG, DILI, Ames, LD50/GHS, reactive metabolite
     │
     ▼
[classify_risk]
     │
     ├── HIGH ──► [hitl_checkpoint] ──approve──► [lead_optimization]
     │                              └─reject──► [report]
     │
     ├── MEDIUM ──► [lead_optimization]
     └── LOW    ──► [lead_optimization]
                          │
                          ▼
                    [generate_report]  ← JSON + Markdown + SQLite audit
```

---

## 12 Tools Built

| Tool | Function | Regulatory Reference |
|------|----------|---------------------|
| validate_smiles | RDKit canonicalization + InChIKey | RDKit MolVS |
| compute_descriptors | MW, LogP, TPSA, HBD/HBA, QED, CNS MPO | Lipinski 1997, Veber 2002 |
| screen_alerts | PAINS (480) + ICH M7 + Brenk alerts | Baell 2010, ICH M7(R2) |
| predict_admet | BBB-Score, Caco-2, CYP, t½, PPB | Gupta 2019, pkCSM |
| predict_toxicity | hERG, DILI, Ames, LD50/GHS, reactive metabolite | FDA DILIrank, ICH S7B |
| generate_fingerprint | ECFP4 (2048-bit) + MACCS keys | Rogers 2010 |
| similarity_search | Tanimoto + applicability domain | Tanimoto 1958 |
| suggest_optimizations | Bioisosteres, LogP reducers, hERG fixes | Meanwell 2011 |
| classify_risk | Evidence aggregation → LOW/MEDIUM/HIGH | IATA, ChatInvent (AZ) |
| lookup_pubchem | PUG-REST / ChEMBL API wrapper | PubChem, ChEMBL |
| estimate_synthesizability | SA Score + retrosynthesis complexity | Ertl 2009 |
| predict_solubility | ESOL model (Delaney 2004) | logS = f(logP, MW, RB, ArRings) |

---

## Installation

```bash
# Minimal (mock mode — no RDKit needed for testing)
pip install scikit-learn numpy pandas

# Full (with RDKit)
conda install -c conda-forge rdkit scikit-learn
pip install numpy pandas

# With API serving
pip install fastapi uvicorn pydantic

# With full LangGraph (production)
pip install langgraph langchain-anthropic langchain-core
```

---

## Usage

### CLI — single compound
```bash
python chemagent_pipeline.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --name "Aspirin"
```

### CLI — demo (3 test compounds)
```bash
python chemagent_pipeline.py --demo
```

### CLI — unit tests
```bash
python chemagent_pipeline.py --test
```

### Python API
```python
from chemagent_pipeline import ChemAgentPipeline

pipeline = ChemAgentPipeline()
state = pipeline.run(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    name="Aspirin",
    auto_approve_hitl=False   # True = skip HITL prompt (batch mode)
)
print(state["risk_level"])   # LOW / MEDIUM / HIGH
print(state["report_path"])  # reports/report_XXXXXXXX.json
```

### Batch processing
```python
compounds = [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
    {"smiles": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "name": "Celecoxib"},
]
results = pipeline.run_batch(compounds, auto_approve=True)
```

### FastAPI server
```bash
python chemagent_pipeline.py --serve --port 8000
# Docs: http://localhost:8000/docs
```

### REST API endpoints
```bash
# Analyze a compound
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O","name":"Aspirin"}'

# Submit HITL decision
curl -X POST http://localhost:8000/hitl/{thread_id} \
  -d '{"decision":"approve","comment":"Reviewed and acceptable"}'

# Get session state
curl http://localhost:8000/session/{thread_id}
```

---

## Full LangGraph Production Migration

Replace the manual orchestrator with real LangGraph (5 key changes):

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. Build graph
workflow = StateGraph(CompoundState)
workflow.add_node("parse_input",       node_parse_input)
workflow.add_node("compute_desc",      node_compute_descriptors)
workflow.add_node("screen_alerts",     node_screen_alerts)
workflow.add_node("predict_admet",     node_predict_admet)
workflow.add_node("predict_toxicity",  node_predict_toxicity)
workflow.add_node("classify_risk",     node_classify_risk)
workflow.add_node("hitl_checkpoint",   node_hitl_checkpoint)
workflow.add_node("lead_optimization", node_lead_optimization)
workflow.add_node("generate_report",   node_generate_report)

# 2. Add edges
workflow.set_entry_point("parse_input")
workflow.add_edge("parse_input", "compute_desc")
# ... (see Stage 9 docstring in chemagent_pipeline.py)

# 3. Conditional edges
workflow.add_conditional_edges("classify_risk", route_after_risk,
    {"hitl":"hitl_checkpoint","optimize":"lead_optimization","report":"generate_report"})

# 4. Compile with checkpointer (required for HITL interrupts)
checkpointer = SqliteSaver.from_conn_string("sessions.db")
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["hitl_checkpoint"]   # pause here for human approval
)

# 5. Run (pauses at hitl_checkpoint if HIGH risk)
state = app.invoke(
    {"input_smiles": smiles, "thread_id": "my-session"},
    config={"configurable": {"thread_id": "my-session"}}
)
# Resume after human decision:
state = app.invoke(
    Command(resume={"hitl_decision": "approve"}),
    config={"configurable": {"thread_id": "my-session"}}
)
```

---

## Why This Pattern vs Simple Script

| Feature | Simple Script | ChemAgent Pro |
|---------|--------------|--------------|
| State persistence | ✗ | ✓ SQLite/Postgres |
| Resume after crash | ✗ | ✓ thread_id |
| Human approval gate | ✗ | ✓ HITL checkpoint |
| Conditional routing | if/else | ✓ LangGraph edges |
| Typed state | ✗ dict | ✓ TypedDict |
| Audit trail | ✗ | ✓ every decision logged |
| REST API | ✗ | ✓ FastAPI |
| Batch processing | ✗ | ✓ |
| Unit tests | ✗ | ✓ 12 tool tests |
| Regulatory compliance | ✗ | ✓ ICH M7, IATA |

---

## References

- ChatInvent (AstraZeneca): Democratising real-world drug discovery through agentic AI, Drug Disc Today 2026
- ChemCrow: Augmenting LLMs with chemistry tools, Bran 2024, Nat Mach Intell  
- ChemGraph: Agentic framework for computational chemistry workflows, Comms Chem 2026
- LangGraph: https://github.com/langchain-ai/langgraph
- ICH M7(R2): Assessment and Control of DNA Reactive Mutagenic Impurities, 2023
- FDA DILIrank / LTKB: https://www.fda.gov/science-research/liver-toxicity-knowledge-base
- OPERA: Mansouri 2019, J Cheminform

---

*ChemAgent Pro | Himanshu Goel | himanshugoel.github.io*
