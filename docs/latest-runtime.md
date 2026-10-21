# Latest retrieval runtime

The current runtime incorporates the semantic-navigation and answer prompts from
`memloopdata_68.75_github_release`. Existing memory construction, hierarchy files,
node representations, and API configuration remain in place. This update applies
to retrieval, promotion, decay, and answer generation after construction.

## Install and run

Install from the repository root, configure the existing `.env.example` settings
for your model and embedding provider, and reuse a hierarchy from the documented
construction workflow:

```bash
pip install -e ".[all]"
memondemand run \
  --method V5 \
  --hierarchy /path/to/hierarchy.jsonl \
  --queries /path/to/queries.parquet \
  --seed 20260702 \
  --top_k_distilled 50 \
  --promotion_budget 20 \
  --out results/latest \
  --resume
```

The query manifest uses `query_id` and `query_text`. Hierarchies retain the existing
newline-delimited JSON node schema; their filename extension is not significant.
The command calls the configured models and may incur provider charges.

## Runtime behavior

- **Semantic navigation v2:** matches conceptual topics as well as literal entity
  names and can descend into multiple plausible clusters. The action cap defaults
  to three; the outer runner retains its separate `--max_steps` limit.
- **Cluster-aware BM25:** retrieves a global candidate pool, moves candidates in
  the navigator-selected cluster to the front, and retains other candidates as
  fallback. This reranks candidates rather than discarding every outside document.
- **Promotion:** combines embedding relevance, uncertainty, expected information
  gain, historical detail utility, and detailed-token cost. The configured model
  chooses promotion decisions, with a deterministic fallback and a relevance-floor
  gate. Decay uses historical utility and recency to remove stale or excess nodes.
- **Answer prompt v2:** requests exhaustive extraction and grounded synthesis from
  the selected evidence. Full source text is the default evidence representation.

Defaults are applied only when the corresponding setting is absent:

| Setting | Default |
| --- | --- |
| `MEMONDEMAND_L0_RETRIEVAL` | `bm25` |
| `MEMONDEMAND_MAX_NAV_STEPS` | `3` |
| `MEMONDEMAND_SKIP_L0_EMBED` | `1`, only where the selected retrieval path permits it |
| `MEMONDEMAND_ANSWER_MODE` | `detailed_full` |
| `MEMONDEMAND_ANSWER_PROMPT` | `v2` |
| `MEMONDEMAND_CONTEXT_FORMAT` | `dual` |
| `MEMONDEMAND_CLUSTER_FILTER` | `1` |
| `MEMONDEMAND_BM25_WIDE_K` | `50` |
| `MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR` | `0.0`, no additional floor |

`MEMLOOP_*` settings from the release are accepted as historical aliases;
`MEMONDEMAND_*` takes precedence when both are present. Existing API credentials,
model aliases, embedding settings, and endpoint configuration are unchanged.

`CONTEXT_FORMAT=dual` identifies hierarchy background as `[CTX]` and source evidence
as `[EVID]`; complete source text receives a `[FULL]` marker. This changes how
selected evidence is presented to align it with the answer prompt. It does not
change the retrieved document identifiers or establish reproduction of a previous
LLM evaluation. Use `MEMONDEMAND_CONTEXT_FORMAT=plain` for the earlier presentation.
The answer prompt independently defaults to `MEMONDEMAND_ANSWER_PROMPT=v2`.
Legacy `V6_DUAL_MEM`, `V7_ENTITY`, and `V61` flags do not silently replace v2;
set `MEMONDEMAND_ANSWER_PROMPT=legacy` explicitly to select their older prompts.
An explicit `MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE` takes precedence over both.

Custom prompt files remain available through
`MEMONDEMAND_NAV_SYSTEM_OVERRIDE_FILE`, `MEMONDEMAND_NAV_USER_OVERRIDE_FILE`, and
`MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE`. The bundled v2 prompts are package
resources under `memondemand/runners/prompts/`.

## Evaluate saved outputs

These helpers operate on saved answers only. Expected document identifiers are
read by the evaluator and are never passed into retrieval:

```bash
python -m memondemand.eval.convert_to_official_format \
  results/latest/answers.jsonl results/latest/official_answers.jsonl
python -m memondemand.eval.compute_evidence_f1 \
  results/latest/answers.jsonl /path/to/official_questions.jsonl
```

The converter writes the official `{question_id, answer, document_ids}` schema;
answer-quality scoring still uses the upstream benchmark evaluator. Evidence
precision, recall, and F1 are macro averages over matched answer rows with nonempty
`expected_doc_ids`; questions absent from the supplied answer file are not added.
These helpers do not invoke a model.

## Optional relevance-floor sweep

The script below runs only when explicitly invoked. It calls the configured
models for every requested floor and can incur provider charges. It reuses the
same hierarchy and query manifest; it does not rebuild memory or perform judging.

```bash
bash scripts/run_floor_sweep.sh \
  --hierarchy /path/to/hierarchy.jsonl \
  --queries /path/to/queries.parquet \
  --output-root results/floor_sweep \
  --floors "0.1 0.3 0.5 0.7 0.9"
```

Equivalent environment variables are `HIERARCHY`, `QUERIES`, `OUTPUT_ROOT`, and
`FLOORS`. `PYTHON_BIN` selects the Python executable. Extra runner arguments can be
supplied after `--`, for example `-- --n_smoke 5`. Each floor writes to a separate
subdirectory and uses the runner's existing `--resume` behavior.
