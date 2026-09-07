"""Keycloak auth for the eval harness.

The hand-rolled suites run as `eval`. Each standard benchmark runs as its
own user (`eval-scifact`, `eval-hotpotqa`, …) so its multi-thousand-doc
corpus stays isolated — file-service vector search has no path filter, it
searches everything the user owns.

- `access_token(user)` — password grant, cached per user until ~30s before expiry.
- `ensure_user(user)` — create/enable + (re)set password via the Admin API,
  idempotent. Returns the user's `sub` (== the file-service `user_id`).
"""
import base64
import json
import time

import httpx

from jarvis_eval.config import settings

_token_cache: dict[str, tuple[str, float]] = {}


def _kc() -> httpx.Client:
    return httpx.Client(base_url=settings.api_base, headers=settings.auth_headers, timeout=20.0)


def access_token(user: str | None = None) -> str:
    user = user or settings.EVAL_USERNAME
    cached = _token_cache.get(user)
    if cached and time.time() < cached[1]:
        return cached[0]
    with _kc() as c:
        r = c.post(settings.token_url, data={
            "grant_type": "password",
            "client_id": settings.KC_CLIENT_ID,
            "client_secret": settings.KC_CLIENT_SECRET,
            "username": user,
            "password": settings.EVAL_PASSWORD,
        })
    if r.status_code != 200:
        raise RuntimeError(
            f"token for {user!r} failed ({r.status_code}): {r.text[:200]}\n"
            "Run `jeval setup` first."
        )
    body = r.json()
    _token_cache[user] = (body["access_token"], time.time() + body.get("expires_in", 300) - 30)
    return _token_cache[user][0]


def user_sub(user: str | None = None) -> str:
    payload = access_token(user).split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


# backwards-compatible aliases used by the hand-rolled suites
def eval_user_sub() -> str:
    return user_sub()


def _admin_token(c: httpx.Client) -> str:
    r = c.post("/realms/master/protocol/openid-connect/token", data={
        "grant_type": "password",
        "client_id": "admin-cli",
        "username": settings.KC_ADMIN_USERNAME,
        "password": settings.KC_ADMIN_PASSWORD,
    })
    if r.status_code != 200:
        raise RuntimeError(
            f"Keycloak admin token failed ({r.status_code}): {r.text[:200]}\n"
            "Set KC_ADMIN_USERNAME / KC_ADMIN_PASSWORD (see jarvis-keycloak-secrets)."
        )
    return r.json()["access_token"]


def ensure_user(user: str | None = None) -> str:
    """Idempotently create + enable `user`, (re)set its password to
    EVAL_PASSWORD. Returns its `sub`."""
    user = user or settings.EVAL_USERNAME
    realm = settings.KC_REALM
    with _kc() as c:
        admin = _admin_token(c)
        h = {"Authorization": f"Bearer {admin}"}

        r = c.get(f"/admin/realms/{realm}/users",
                  params={"username": user, "exact": "true"}, headers=h)
        r.raise_for_status()
        found = r.json()

        if found:
            uid = found[0]["id"]
            c.put(f"/admin/realms/{realm}/users/{uid}", headers=h,
                  json={"enabled": True, "emailVerified": True}).raise_for_status()
        else:
            cr = c.post(f"/admin/realms/{realm}/users", headers=h, json={
                "username": user, "enabled": True, "emailVerified": True,
                "email": f"{user}@example.test", "firstName": "Eval", "lastName": "Bot",
            })
            if cr.status_code not in (201, 409):
                raise RuntimeError(f"create {user!r} failed ({cr.status_code}): {cr.text[:200]}")
            r = c.get(f"/admin/realms/{realm}/users",
                      params={"username": user, "exact": "true"}, headers=h)
            r.raise_for_status()
            uid = r.json()[0]["id"]

        c.put(f"/admin/realms/{realm}/users/{uid}/reset-password", headers=h, json={
            "type": "password", "value": settings.EVAL_PASSWORD, "temporary": False,
        }).raise_for_status()

    _token_cache.pop(user, None)
    return uid


def ensure_eval_user() -> str:  # kept for the hand-rolled `jeval setup` path
    return ensure_user()
