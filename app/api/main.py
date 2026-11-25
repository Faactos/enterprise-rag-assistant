"""FastAPI surface.

The service and every model are built during startup, not on first request.
Lazy loading made the first /ask download ~1GB of reranker weights inside the
request and time the client out -- a cold start belongs in the boot, where it is
visible, not in a user's first question.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import auth
from ..config import settings
from ..generation.llm import LLMError
from ..service import AccessDenied, RagService

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@lru_cache(maxsize=1)
def service() -> RagService:
    return RagService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the service and warm every model at boot.

    The reranker in particular is ~1GB; loading it here means the process is
    slow to start and fast to answer, rather than the reverse.
    """
    svc = service()
    svc.embedder.warm()
    if settings.use_reranker:
        from ..retrieval.reranker import _cross_encoder

        _cross_encoder(settings.rerank_model)
    yield


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Document-grounded question answering with page-level citations.",
    version="1.0.0",
    lifespan=lifespan,
)


# -- schemas ---------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(..., examples=["me@example.com"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    doc_ids: list[str] | None = None


class ShareRequest(BaseModel):
    email: str


# -- auth dependency -------------------------------------------------------


def current_user(authorization: str | None = Header(default=None)) -> auth.User:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    user = auth.authenticate(token)
    if user is None:
        raise HTTPException(401, "Missing or invalid bearer token.")
    return user


# -- routes ----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "hybrid": settings.use_hybrid,
        "reranker": settings.use_reranker,
    }


@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest) -> dict:
    try:
        user, token = auth.register(req.email)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # The token is returned exactly once; only its hash is stored.
    return {"user_id": user.user_id, "email": user.email, "token": token}


@app.post("/documents", status_code=201)
def upload(
    file: UploadFile = File(...), user: auth.User = Depends(current_user)
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        staged = Path(tmp.name)
    try:
        result = service().ingest(user.user_id, staged, title=file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        staged.unlink(missing_ok=True)
    return result.to_dict()


@app.get("/documents")
def list_documents(user: auth.User = Depends(current_user)) -> dict:
    return {"documents": service().list_documents(user.user_id)}


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, user: auth.User = Depends(current_user)) -> None:
    try:
        service().delete_document(user.user_id, doc_id)
    except KeyError as exc:
        raise HTTPException(404, "No such document.") from exc
    except AccessDenied as exc:
        raise HTTPException(403, str(exc)) from exc


@app.post("/documents/{doc_id}/share")
def share_document(
    doc_id: str, req: ShareRequest, user: auth.User = Depends(current_user)
) -> dict:
    try:
        return service().share_document(user.user_id, doc_id, req.email)
    except KeyError as exc:
        raise HTTPException(404, "No such document or user.") from exc
    except AccessDenied as exc:
        raise HTTPException(403, str(exc)) from exc


@app.post("/ask")
def ask(req: AskRequest, user: auth.User = Depends(current_user)) -> dict:
    try:
        return service().ask(user.user_id, req.question, req.doc_ids).to_dict()
    except LLMError as exc:
        # The provider failed, not this service. A 500 here would send the
        # reader looking for a bug in the retrieval code.
        raise HTTPException(502, f"Generation provider error: {exc}") from exc


@app.post("/retrieve")
def retrieve(req: AskRequest, user: auth.User = Depends(current_user)) -> dict:
    passages = service().retrieve(user.user_id, req.question, req.doc_ids)
    return {"passages": [p.to_dict() for p in passages]}
