VENV := .venv
PIP := $(VENV)/bin/pip
JEVAL := $(VENV)/bin/jeval

NS ?= jarvis
FILE_SVC_LOCAL_PORT ?= 18002

.PHONY: install setup port-forward stop-port-forward seed-all eval eval-retrieval report baseline clean

install:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e .

# Create/enable the eval + eval-<benchmark> Keycloak users (idempotent).
setup:
	$(JEVAL) setup

# kubectl port-forward file-service in the background (needed by seed + all
# retrieval scoring). Writes the PID to .pf.pid.
port-forward:
	@kubectl -n $(NS) port-forward svc/file-service $(FILE_SVC_LOCAL_PORT):8000 > .pf.log 2>&1 & echo $$! > .pf.pid
	@sleep 2 && echo "file-service -> http://localhost:$(FILE_SVC_LOCAL_PORT)  (pid $$(cat .pf.pid))"

stop-port-forward:
	@[ -f .pf.pid ] && kill $$(cat .pf.pid) 2>/dev/null && rm -f .pf.pid .pf.log || true

# One-off: download + upload + embed every non-gated benchmark corpus (~40 min).
seed-all:
	$(JEVAL) seed beir_scifact
	$(JEVAL) seed beir_nfcorpus
	$(JEVAL) seed hotpotqa

# Score all benchmarks against the committed baseline.
eval:
	$(JEVAL) run --suite all
	$(JEVAL) report --baseline --fail-on-regression

# Cheap CI gate: the two BEIR retrieval suites only (no agent, near-free).
eval-retrieval:
	$(JEVAL) run --suite beir_scifact
	$(JEVAL) run --suite beir_nfcorpus
	$(JEVAL) report --baseline --fail-on-regression

report:
	$(JEVAL) report --baseline

# Promote the newest results/ run into datasets/baseline.json.
baseline:
	$(JEVAL) baseline

clean:
	rm -rf $(VENV) .pf.pid .pf.log
