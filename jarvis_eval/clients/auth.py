"""Keycloak auth for the eval harness.

- `access_token()` — password grant for the eval user, cached until ~30s
  before expiry.
- `ensure_eval_user()` — create/enable the `eval` user and (re)set its
  password via the Admin API, idempotent. Returns the user's `sub`, which
  is also the `user_id` the backend hands to file-service and the agent
  tools, so the seeded corpus and the agent runs line up.
"""
import base64
import json
import time

import httpx

from jarvis_eval.config import settings

_token_cache: dict = {"value": None, "exp": 0.0}


def _kc() -> httpx.Client:
    return httpx.Client(base_url=settings.api_base, headers=settings.auth_headers, timeout=20.0)


def access_token() -> str:
    now = time.time()
    if _token_cache["value"] and now < _token_cache["exp"]:
        return _token_cache["value"]
    with _kc() as c:
        r = c.post(settings.token_url, data={
            "grant_type": "password",
            "client_id": settings.KC_CLIENT_ID,
            "client_secret": settings.KC_CLIENT_SECRET,
            "username": settings.EVAL_USERNAME,
            "password": settings.EVAL_PASSWORD,
        })
    if r.status_code != 200:
        raise RuntimeError(
            f"eval-user token failed ({r.status_code}): {r.text[:200]}\n"
            "Run `jeval setup` first to create the eval user."
        )
    body = r.json()
    _token_cache["value"] = body["access_token"]
    _token_cache["exp"] = now + body.get("expires_in", 300) - 30
    return _token_cache["value"]


def eval_user_sub() -> str:
    """The eval user's Keycloak id, read straight from its JWT `sub`."""
    tok = access_token()
    payload = tok.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


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


def ensure_eval_user() -> str:
    """Idempotently create + enable the eval user, (re)set its password.
    Returns its `sub`."""
    realm = settings.KC_REALM
    with _kc() as c:
        admin = _admin_token(c)
        h = {"Authorization": f"Bearer {admin}"}

        r = c.get(f"/admin/realms/{realm}/users",
                  params={"username": settings.EVAL_USERNAME, "exact": "true"}, headers=h)
        r.raise_for_status()
        found = r.json()

        if found:
            uid = found[0]["id"]
            c.put(f"/admin/realms/{realm}/users/{uid}", headers=h,
                  json={"enabled": True, "emailVerified": True}).raise_for_status()
        else:
            cr = c.post(f"/admin/realms/{realm}/users", headers=h, json={
                "username": settings.EVAL_USERNAME,
                "enabled": True,
                "emailVerified": True,
                "email": f"{settings.EVAL_USERNAME}@example.test",
                "firstName": "Eval",
                "lastName": "Bot",
            })
            if cr.status_code not in (201, 409):
                raise RuntimeError(f"create eval user failed ({cr.status_code}): {cr.text[:200]}")
            r = c.get(f"/admin/realms/{realm}/users",
                      params={"username": settings.EVAL_USERNAME, "exact": "true"}, headers=h)
            r.raise_for_status()
            uid = r.json()[0]["id"]

        c.put(f"/admin/realms/{realm}/users/{uid}/reset-password", headers=h, json={
            "type": "password", "value": settings.EVAL_PASSWORD, "temporary": False,
        }).raise_for_status()

    # bust the token cache so the next access_token() picks up the new creds
    _token_cache["value"], _token_cache["exp"] = None, 0.0
    return uid
