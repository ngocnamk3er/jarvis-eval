# Deployment runbook

## Bring the test stack up after a reboot
    minikube start -p minikube

ArgoCD re-syncs everything. Two things break on every node restart:

1. Keycloak CrashLoopBackOff on "UnknownHostException: keycloak-postgres"
   — it races CoreDNS. Fix:
       kubectl rollout restart deployment/keycloak -n jarvis

2. The --insecure-registry flag for host.minikube.internal:5050 is lost, so
   Jenkins-built images fail ErrImagePull. Fix: sed it back into
   /lib/systemd/system/docker.service inside the node and restart docker.

## Deploy jarvis-file-service (no Jenkins job yet)
Build locally, load under a NEW commit-SHA tag (minikube image load no-ops
on an existing tag), bump file-service/overlays/test/kustomization.yaml,
push jarvis-deploy, let ArgoCD sync.
