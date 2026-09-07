# Why the DB is the source of truth, not the object store

Object storage (MinIO/S3) is great at exactly one thing: storing bytes
durably and cheaply at scale. Everything about what those bytes *mean* —
the name, the owner, where it sits in the tree — is business logic that
needs transactions, constraints, and fast filtered queries. That's the
database's job.

Concretely in file_nodes:
- UNIQUE(user_id, parent_id, name) — no duplicate names in a folder
- parent_id self-FK with ON DELETE CASCADE — delete a folder, the subtree goes
- indexed lookups by user_id / parent_id

The MinIO object key is an opaque {user_id}/{file_id} — never the path —
so a rename or move is one UPDATE and never touches storage.

Same split shows up everywhere: Git (commit graph vs blobs), Docker
(manifest vs layers), streaming services (metadata DB vs video on a CDN).
