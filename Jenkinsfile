pipeline {
  agent any

  // Manual ("Build Now") or nightly. Not triggered on push — a run costs
  // OpenRouter credit and takes a few minutes.
  triggers { cron('H 3 * * *') }

  options {
    timeout(time: 30, unit: 'MINUTES')
    disableConcurrentBuilds()
  }

  environment {
    // A "Secret file" credential holding the .env for the harness:
    // INTERNAL_API_KEY, KC_ADMIN_USERNAME/PASSWORD, HF_TOKEN (for GAIA),
    // and INGRESS_IP if it isn't 192.168.49.2. See README.
    EVAL_ENV = credentials('jarvis-eval-env')
  }

  stages {
    stage('Checkout') { steps { checkout scm } }

    stage('Install') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install -q --upgrade pip
          pip install -q -e .
          cp "$EVAL_ENV" .env
        '''
      }
    }

    stage('Port-forward file-service') {
      steps {
        sh '''
          nohup kubectl -n jarvis port-forward svc/file-service 18002:8000 > pf.log 2>&1 &
          echo $! > pf.pid
          for i in $(seq 1 15); do curl -sf localhost:18002/api/v1/health >/dev/null && break; sleep 1; done
        '''
      }
    }

    // Assumes the benchmark corpora are already seeded (one-off `make
    // seed-all`, ~40 min). This job just re-scores + gates — the BEIR
    // retrieval suites are near-free; add `hotpotqa` for the agent path.
    stage('Setup') {
      steps { sh '. .venv/bin/activate && jeval setup' }
    }

    stage('Run BEIR retrieval') {
      steps {
        sh '''
          . .venv/bin/activate
          jeval run --suite beir_scifact
          jeval run --suite beir_nfcorpus
        '''
      }
    }

    stage('Report + regression gate') {
      steps {
        sh '. .venv/bin/activate && jeval report --baseline --fail-on-regression'
      }
    }
  }

  post {
    always {
      sh 'test -f pf.pid && kill $(cat pf.pid) || true'
      archiveArtifacts artifacts: 'results/**/report.md, results/**/results.json', allowEmptyArchive: true
    }
  }
}
