"""Bearer-token auth.

Tokens are shown once and stored only as a SHA-256 hash, so a leaked database
cannot be replayed against the API.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from . import db


@dataclass(frozen=True)
class User:
    user_id: str
    email: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register(email: str) -> tuple[User, str]:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email address is required.")
    token = secrets.token_urlsafe(32)
    user = User(user_id=str(uuid.uuid4()), email=email)
    with db.connect() as conn:
        existing = conn.execute(
            "SELECT user_id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            # Re-registering rotates the token rather than failing: the token is
            # the only credential, and there is nothing else to recover it with.
            conn.execute(
                "UPDATE users SET token_hash = ? WHERE email = ?", (_hash(token), email)
            )
            return User(existing["user_id"], email), token
        conn.execute(
            "INSERT INTO users (user_id, email, token_hash, created_at) VALUES (?,?,?,?)",
            (user.user_id, email, _hash(token), datetime.now(timezone.utc).isoformat()),
        )
    return user, token


def authenticate(token: str | None) -> User | None:
    if not token:
        return None
    with db.connect() as conn:
        row = conn.execute(
            "SELECT user_id, email FROM users WHERE token_hash = ?", (_hash(token),)
        ).fetchone()
    return User(row["user_id"], row["email"]) if row else None
