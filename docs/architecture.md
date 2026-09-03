# Architecture

MemOnDemand is organized as a local-first retrieval package. It keeps data
preparation, retrieval control, answer generation, and evaluation in separate
layers so teams can replace storage, model gateways, and orchestration without
rewriting the full pipeline.

## Runtime Layers

| Layer | Purpose | Primary modules |
| --- | --- | --- |
| Data plane | Read evidence manifests, build hierarchy nodes, and persist local artifacts. | `memondemand.data`, `memondemand.methods.dual_node` |
| Retrieval plane | Search L0 evidence, navigate hierarchy nodes, promote useful detail, and assemble answer context. | `memondemand.methods`, `memondemand.runners` |
| Operations plane | Load general gateway configuration, track token use, run evaluations, and expose CLI entry points. | `memondemand.core`, `memondemand.eval`, `memondemand.cli` |

## Query Lifecycle

1. Load hierarchy and query manifest.
2. Retrieve global L0 and high-level hierarchy candidates.
3. Navigate only the hierarchy branches relevant to the query.
4. Update ranking and bounded cross-query state through on-demand promotion.
5. Resolve selected nodes to detailed L0 evidence under the answer budget.
6. Write answer, citation, token, and evaluation artifacts to the run directory.

Promotion affects managed retrieval state across queries. Detailed evidence loading is a
separate, budgeted operation, and distilled text is not treated as answer evidence.

## Extension Points

| Extension | How to customize |
| --- | --- |
| Model gateway | Configure general chat and embedding endpoints with environment variables. |
| Retrieval backend | Use BM25 by default; extend index code under `memondemand.methods` for alternative retrieval. |
| Storage | Keep manifests in local parquet/JSONL files, or wrap reads and writes before calling the CLI modules. |
| Evaluation | Add custom metrics under `memondemand.eval` and expose them through the CLI. |
| Orchestration | Run the CLI from Airflow, cron, batch systems, notebooks, or service workers. |

## Artifact Boundaries

MemOnDemand assumes private corpora, generated answers, caches, and logs are local
runtime artifacts. They are intentionally ignored by Git. Public commits should
contain package code, small fixtures, docs, and reproducible command surfaces,
not customer data or generated benchmark output.
