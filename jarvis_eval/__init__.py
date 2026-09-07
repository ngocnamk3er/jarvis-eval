"""jarvis-eval — a black-box benchmark harness for the Jarvis agent.

It doesn't test Jarvis's internals; it drives the *deployed* test cluster
over HTTP, exactly as a user would, and scores the results with standard
benchmark metrics.

Pipeline (one benchmark):

  jeval setup            create a Keycloak user for it            (auth.py)
  jeval seed <bench>     HF dataset -> upload corpus -> embed      (hf.py, files.py, benchmarks/*.seed)
  jeval run --suite ...  query file-service search / run the agent (files.py, chat.py, benchmarks/*.run)
                         -> score with nDCG / recall / EM / F1     (metrics.py)
                         -> results/<ts>/{raw,results}.json + report.md   (report.py)
  jeval report           compare to datasets/baseline.json + gate  (report.py)
  jeval baseline         freeze the current scores as the new bar

Benchmarks: BEIR SciFact / NFCorpus (retrieval), HotpotQA (multi-hop RAG),
GAIA (agent tasks — gated).
"""

__version__ = "0.2.0"
