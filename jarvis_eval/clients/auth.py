"""Keycloak authentication.

Two jobs:

* `access_token(user)` — log in as an eval user via the OIDC *password
  grant* (username + password -> JWT), cached per user until ~30s before it
  expires. This token is the bearer for jarvis-backend calls.

* `ensure_user(user)` — make sure that user exists in Keycloak, via the
  *Admin API* (which itself needs a master-realm admin login). Idempotent:
  creates if missing, enables + resets the password if present. This is all
  `jeval setup` does.

Each benchmark runs as its own user (`eval-beir-scifact`, `eval-hotpotqa`,
…). Reason: file-service vector search has no path filter — it searches
everything a user owns — so separate users keep the 5k SciFact docs from
polluting HotpotQA's results.

A user's Keycloak `id` (returned by ensure_user) equals its JWT `sub`,
which is also the `user_id` file-service and the agent key data by. So all
three line up automatically.
"""
import base64
import json
import time

import httpx

from jarvis_eval.config import settings

# username -> (jwt, expires_at_epoch). Avoids a token round-trip per request.
_token_cache: dict[str, tuple[str, float]] = {}


def _kc() -> httpx.Client:
    """An httpx client aimed at Keycloak (ingress IP + `Host: auth.jarvis.local`)."""
    return httpx.Client(base_url=settings.api_base, headers=settings.auth_headers, timeout=20.0)


def access_token(user: str | None = None) -> str:
    """A valid bearer JWT for `user` (default: the base `eval` user).
    Cached; refetched only when expired."""
    user = user or settings.EVAL_USERNAME
    cached = _token_cache.get(user)
    if cached and time.time() < cached[1]:
        return cached[0]
    with _kc() as c:
        r = c.post(settings.token_url, data={          # OIDC password grant
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
    # cache with a 30s safety margin before the real expiry
    _token_cache[user] = (body["access_token"], time.time() + body.get("expires_in", 300) - 30)
    return _token_cache[user][0]


def user_sub(user: str | None = None) -> str:
    """The `sub` claim of the user's JWT — i.e. their Keycloak id, which is
    the `user_id` file-service wants. Decoded from the token's middle segment
    (no signature check needed, we just made the token)."""
    payload = access_token(user).split(".")[1]
    payload += "=" * (-len(payload) % 4)             # pad to a multiple of 4 for base64
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


def _admin_token(c: httpx.Client) -> str:
    """A master-realm admin token (client `admin-cli`, KC_ADMIN_* creds) —
    needed to call the Keycloak Admin API in ensure_user()."""
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
    """Create + enable `user` and (re)set its password to EVAL_PASSWORD.
    Idempotent — safe to call every run. Returns the user's `sub`."""
    user = user or settings.EVAL_USERNAME
    realm = settings.KC_REALM
    with _kc() as c:
        h = {"Authorization": f"Bearer {_admin_token(c)}"}

        # already there?
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
            if cr.status_code not in (201, 409):        # 409 = race, someone else created it
                raise RuntimeError(f"create {user!r} failed ({cr.status_code}): {cr.text[:200]}")
            # the create response has no body; look the id up
            r = c.get(f"/admin/realms/{realm}/users",
                      params={"username": user, "exact": "true"}, headers=h)
            r.raise_for_status()
            uid = r.json()[0]["id"]

        c.put(f"/admin/realms/{realm}/users/{uid}/reset-password", headers=h, json={
            "type": "password", "value": settings.EVAL_PASSWORD, "temporary": False,
        }).raise_for_status()

    _token_cache.pop(user, None)    # creds may have changed -> drop any cached token
    return uid
