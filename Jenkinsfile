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
    // A "Secret file" credential holding the .env for the eval harness:
    // OPENROUTER_API_KEY, INTERNAL_API_KEY, KC_ADMIN_USERNAME/PASSWORD,
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

    stage('Setup + seed') {
      steps {
        sh '''
          . .venv/bin/activate
          jeval setup
          jeval seed
        '''
      }
    }

    stage('Run smoke suite') {
      steps {
        sh '''
          . .venv/bin/activate
          jeval run --suite smoke
        '''
      }
    }

    stage('Report + regression gate') {
      steps {
        sh '''
          . .venv/bin/activate
          jeval report --baseline --fail-on-regression
        '''
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
