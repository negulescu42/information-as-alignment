# § C4 — Lifecycle Comparison: IBF vs kNN vs RAG

**Engine:** `2.1-history_gate+xcontext_flag`  
**Run mode:** `paper`  
**Records:** `30` (CounterFact-style synthetic)

## Headline

- WITHIN_TOLERANCE: `True`
- IBF native: direct=1.000 loc=1.000 rev=1.000 rem=1.000 rb=1.000
- kNN oracle: direct=1.000 loc=0.000 rev=1.000 rem=1.000 rb=1.000
- RAG oracle: direct=0.967 loc=0.000 rev=0.967 rem=1.000 rb=0.967

## Lifecycle metrics table

| Dimension | IBF (native) | kNN (oracle) | RAG (oracle) |
|---|---:|---:|---:|
| direct | 1.000 | 1.000 | 0.967 |
| locality | 1.000 | 0.000 | 0.000 |
| revise | 1.000 | 1.000 | 0.967 |
| remove | 1.000 | 1.000 | 1.000 |
| rollback | 1.000 | 1.000 | 0.967 |

## Operational burden (architectural distinction)

- **IBF:** `native` — install/revise/remove/rollback are intrinsic δR-field dynamics (write / contradict / locally melt / re-write).
- **kNN:** `oracle-maintained (manual entry add/remove per op)` — every lifecycle op requires explicit entry add/remove by an external oracle.
- **RAG:** `oracle-maintained (manual passage add/remove per op)` — every lifecycle op requires explicit passage add/remove by an external oracle, and answer quality depends on passage retrieval + downstream LM behaviour.

## Interpretation

IBF stores modification sites whose thermodynamic state changes through interaction (crystallisation, dissolution, broadcast-rights). kNN stores static support points; RAG stores text traces. The native vs oracle-maintained operational burden distinction instantiates the architectural argument from foundational § 8.2.