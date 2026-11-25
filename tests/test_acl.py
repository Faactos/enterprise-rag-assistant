"""Access control, enforced in the data layer.

The scope these return is what retrieval is *filtered by*, so a document missing
from this list can never reach a reranker, a prompt, or a citation.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app import auth, db


def make_doc(doc_id: str, owner_id: str, title: str = "doc") -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, owner_id, title, filename, pages, chunks, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (doc_id, owner_id, title, f"{doc_id}.md", 1, 1, datetime.now(timezone.utc).isoformat()),
        )


def share(doc_id: str, user_id: str) -> None:
    with db.connect() as conn:
        conn.execute("INSERT INTO shares (doc_id, user_id) VALUES (?,?)", (doc_id, user_id))


def test_a_user_sees_only_their_own_documents(temp_db):
    alice, _ = auth.register("alice@example.com")
    bob, _ = auth.register("bob@example.com")
    make_doc("d-alice", alice.user_id)
    make_doc("d-bob", bob.user_id)

    assert db.readable_doc_ids(alice.user_id) == ["d-alice"]
    assert db.readable_doc_ids(bob.user_id) == ["d-bob"]


def test_sharing_grants_read_access_without_transferring_ownership(temp_db):
    alice, _ = auth.register("alice@example.com")
    bob, _ = auth.register("bob@example.com")
    make_doc("d-alice", alice.user_id)
    share("d-alice", bob.user_id)

    assert "d-alice" in db.readable_doc_ids(bob.user_id)
    with db.connect() as conn:
        owner = conn.execute(
            "SELECT owner_id FROM documents WHERE doc_id = ?", ("d-alice",)
        ).fetchone()["owner_id"]
    assert owner == alice.user_id


def test_a_user_with_no_documents_has_an_empty_scope(temp_db):
    user, _ = auth.register("nobody@example.com")
    assert db.readable_doc_ids(user.user_id) == []


def test_an_unknown_user_id_sees_nothing(temp_db):
    # Failing open here would hand an unauthenticated caller the whole corpus.
    assert db.readable_doc_ids("not-a-real-user") == []


def test_owned_and_shared_documents_are_not_duplicated(temp_db):
    alice, _ = auth.register("alice@example.com")
    make_doc("d-alice", alice.user_id)
    share("d-alice", alice.user_id)  # owner also listed as a share
    assert db.readable_doc_ids(alice.user_id) == ["d-alice"]


# -- tokens ----------------------------------------------------------------


def test_the_raw_token_is_never_stored(temp_db):
    _, token = auth.register("alice@example.com")
    with db.connect() as conn:
        stored = conn.execute("SELECT token_hash FROM users").fetchone()["token_hash"]
    assert stored != token
    assert token not in stored


def test_a_valid_token_authenticates_to_its_owner(temp_db):
    user, token = auth.register("alice@example.com")
    assert auth.authenticate(token).user_id == user.user_id


@pytest.mark.parametrize("token", [None, "", "not-a-token"])
def test_a_bad_token_authenticates_to_nobody(temp_db, token):
    auth.register("alice@example.com")
    assert auth.authenticate(token) is None


def test_re_registering_rotates_the_token_and_keeps_the_user_id(temp_db):
    first, old = auth.register("alice@example.com")
    second, new = auth.register("alice@example.com")
    assert first.user_id == second.user_id
    assert auth.authenticate(new) is not None
    assert auth.authenticate(old) is None  # the old credential stops working


def test_registration_rejects_a_malformed_email(temp_db):
    with pytest.raises(ValueError):
        auth.register("not-an-email")
