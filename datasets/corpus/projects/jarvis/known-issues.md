# Known issues

- No Jenkins job for jarvis-file-service — manual build + minikube image load.
- MinIO Content-Disposition filename is set at upload, not refreshed on
  rename. App downloads are still correct (name comes from Postgres).
- No transactional consistency between the MinIO write and the Postgres
  commit on upload — a crash in between orphans a MinIO object.
- Old `conversations` / `subagent_traces` tables still sit in the `jarvis`
  database, unused since the Chapter 2 split.
- search_files over a tiny corpus gives weak/noisy scores — expected.
