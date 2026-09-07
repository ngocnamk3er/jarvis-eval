VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
JEVAL := $(VENV)/bin/jeval

NS ?= jarvis
FILE_SVC_LOCAL_PORT ?= 18002

.PHONY: install setup seed port-forward stop-port-forward eval eval-smoke report baseline clean

install:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e .

# Create/enable the `eval` Keycloak user (idempotent).
setup:
	$(JEVAL) setup

# kubectl port-forward file-service in the background (needed by `seed` and
# the retrieval suite). Writes the PID to .pf.pid.
port-forward:
	@kubectl -n $(NS) port-forward svc/file-service $(FILE_SVC_LOCAL_PORT):8000 > .pf.log 2>&1 & echo $$! > .pf.pid
	@sleep 2 && echo "file-service -> http://localhost:$(FILE_SVC_LOCAL_PORT)  (pid $$(cat .pf.pid))"

stop-port-forward:
	@[ -f .pf.pid ] && kill $$(cat .pf.pid) 2>/dev/null && rm -f .pf.pid .pf.log || true

# Wipe + re-upload datasets/corpus to the eval user's workspace, wait for indexing.
seed:
	$(JEVAL) seed

# Full run of every suite, then a report against the committed baseline.
eval:
	$(JEVAL) run --suite all
	$(JEVAL) report --baseline

# Quick CI gate: 3 cases per suite, fail the build on a metric regression.
eval-smoke:
	$(JEVAL) run --suite smoke
	$(JEVAL) report --baseline --fail-on-regression

report:
	$(JEVAL) report --baseline

# Promote the newest results/ run to datasets/baseline.json.
baseline:
	$(JEVAL) baseline

clean:
	rm -rf $(VENV) .judge_cache.json .pf.pid .pf.log
