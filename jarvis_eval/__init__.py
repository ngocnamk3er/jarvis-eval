"""jarvis-eval — a black-box benchmark harness for the Jarvis agent.

Runs standard public benchmarks (BEIR SciFact/NFCorpus, HotpotQA, GAIA)
against the deployed test cluster. Each benchmark's corpus is downloaded
from HuggingFace, uploaded into its own dedicated `eval-<name>` Keycloak
file workspace and embedded with bge-m3, then scored with the standard
metrics (nDCG@10 / recall@k for retrieval, EM/F1 for QA) so the numbers
compare directly to published SOTA.
"""

__version__ = "0.2.0"
