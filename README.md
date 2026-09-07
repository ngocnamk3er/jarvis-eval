# jarvis-eval

Black-box evaluation & benchmarking for the Jarvis agent. Runs against the
**deployed test cluster** as a dedicated `eval` Keycloak user, seeds a
versioned corpus into that user's file workspace, and scores three suites.

| suite | what it measures | how |
|---|---|---|
| `retrieval` | recall@{1,3,5,10}, precision@5, MRR@10, nDCG@10, top-1 score | file-service `/search/vector` straight, gold = file path(s). No LLM — fast + cheap. |
| `rag_qa` | answer correctness, groundedness, citation recall + latency / tokens / $ | full agent run through `/api/v1/chat/stream`, graded by an LLM judge |
| `agent_tasks` | task success rate, tool-choice quality + efficiency | full agent run; per-case deterministic check (`present_file` / `final_contains` / `sandbox_file`) or judge rubric |

## Quick start

```bash
make install                       # .venv + editable install (gives you `jeval`)
cp .env.example .env               # fill OPENROUTER_API_KEY + INTERNAL_API_KEY
jeval setup                        # create/enable the eval Keycloak user (idempotent)
make port-forward                  # kubectl port-forward svc/file-service 18002:8000
jeval seed                         # wipe + upload datasets/corpus, wait for indexing
jeval run --suite smoke            # 3 cases/suite — quick check
jeval run --suite all --repeats 3  # full run
jeval baseline                     # promote the newest run to datasets/baseline.json
make stop-port-forward
```

`INTERNAL_API_KEY` and the Keycloak admin password come from cluster secrets:

```bash
kubectl -n jarvis get secret jarvis-secrets \
  -o jsonpath='{.data.INTERNAL_API_KEY}' | base64 -d ; echo
kubectl -n jarvis get secret jarvis-keycloak-secrets \
  -o jsonpath='{.data.KC_BOOTSTRAP_ADMIN_PASSWORD}' | base64 -d ; echo
```

## Datasets

- `datasets/corpus/` — the ~18-doc fixture workspace (versioned; grow it freely).
- `datasets/{retrieval,rag_qa,agent_tasks}.jsonl` — one case per line.
- `datasets/baseline.json` — committed aggregate metrics the CI gate compares against.

Adding a case = one JSONL line. Case schemas are documented at the top of each
`jarvis_eval/runners/*.py`.

## Regression gate

`jeval report --baseline --fail-on-regression` exits non-zero if, versus
`baseline.json`:

- `retrieval` recall@5 drops more than 0.03
- `rag_qa` answer_correctness drops more than 0.30
- `agent_tasks` success rate drops more than 0.05

After a deliberate improvement, re-run `--suite all` and `jeval baseline` to
move the bar up.

## Cost / determinism

`smoke` ≈ $0.10, `all --repeats 3` ≈ $1–2 of OpenRouter credit (runner
`deepseek/deepseek-v4-flash`, judge `deepseek/deepseek-v4-pro`). The backend
already runs at `temperature=0`; `--repeats` averages out the residual API
nondeterminism. Judge verdicts are cached in `.judge_cache.json` so
`report` / partial re-runs don't re-pay.

## Jenkins job

Not auto-triggered (slow + costs credit). Create it once:

1. **New Item → Pipeline**, name `jarvis-eval`.
2. **Pipeline → Definition: Pipeline script from SCM**, SCM Git,
   URL `http://gitlab:8929/root/jarvis-eval.git`, branch `main`,
   Script Path `Jenkinsfile`.
3. Add a **Secret file** credential with id `jarvis-eval-env` containing a
   filled `.env` (at least `OPENROUTER_API_KEY`, `INTERNAL_API_KEY`,
   `KC_ADMIN_PASSWORD`; `INGRESS_IP` if not `192.168.49.2`).
4. The Jenkins agent needs `kubectl` with a kubeconfig that can reach the
   `jarvis` namespace (for the file-service port-forward).
5. "Build periodically" `H 3 * * *` is set in the `Jenkinsfile`; also
   runnable manually. `report.md` is archived on every build; the build
   goes red on a regression.

## Layout

```
jarvis_eval/
  config.py            settings (.env)
  dataset.py           JSONL loaders
  trace.py             RunTrace — the unit every runner scores; MODEL_PRICING
  metrics.py           recall@k / MRR / nDCG + aggregate helpers
  judge.py             LLM-as-judge over OpenRouter, disk-cached
  report.py            aggregate -> results.json + report.md, baseline diff, gate
  cli.py               `jeval`
  clients/
    auth.py            eval-user token + Keycloak Admin API
    conversations.py   create / delete / messages / sandbox-file
    files.py           seed via backend proxy; search via file-service direct
    chat.py            SSE driver: run_agent(prompt) -> RunTrace
  runners/
    retrieval.py  rag_qa.py  agent_tasks.py
datasets/   results/   Makefile   Jenkinsfile
```

Part of a larger arc (Binance "Agentic-RAG Engineer" JD): eval first, then a
feedback loop, then agentic-RAG improvements (query decomposition, rerank,
retrieve-reflect-refine), then long-term memory — each measured here.
