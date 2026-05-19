# server.py  — run with: uvicorn server:app --port 8001

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
from pos_weighted_rag import RAGPipeline, PipelineConfig
from langchain_core.documents import Document
import logging
import time


from dotenv import load_dotenv
import os

# 1. Load the environment variables from the .env file
load_dotenv() 

# (Optional) Verify it loaded correctly in your terminal
logger = logging.getLogger(__name__)
if not os.environ.get("GOOGLE_API_KEY"):
    logger.warning("WARNING: GOOGLE_API_KEY is missing from the environment!")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
from pos_weighted_rag import load_and_chunk_docs

app = FastAPI()

# ── Bootstrap with sample docs (replace with DB load) ──
_docs = []
#Document(page_content="MLForge uses a DVC clone mechanism to ensure strict data versioning across all distributed nodes.", metadata={"doc_type": "architecture"}),
   # Document(page_content="To trigger deployment automation in the cluster, push the updated model artifacts to the registry.", metadata={"doc_type": "operations"}),
    #Document(page_content="Standard data visualization tools can connect to the platform via the REST API.", metadata={"doc_type": "integrations"}),
   # Document(page_content="Model pipelining requires defining sequential execution steps in the YAML configuration.", metadata={"doc_type": "architecture"}),

_docs     = load_and_chunk_docs("./knowledge_base", chunk_size=800)
_cfg      = PipelineConfig()
_pipeline = RAGPipeline(_docs, _cfg)


# ── Request / Response schemas ──

class QueryRequest(BaseModel):
    query:         str
    top_k:         Optional[int]  = None   # override config at runtime
    dense_weight:  Optional[float] = None
    sparse_weight: Optional[float] = None

class DocResult(BaseModel):
    content:  str
    metadata: dict

class QueryResponse(BaseModel):
    query:   str
    results: List[DocResult]


class IngestRequest(BaseModel):
    content:  str
    metadata: dict = {}


# ── Endpoints ──

@app.post("/retrieve", response_model=QueryResponse)
def retrieve(req: QueryRequest):
    try:
        # Allow per-request overrides without mutating global config
        if req.top_k:
            _pipeline.cfg.top_k_rerank = req.top_k
        if req.dense_weight:
            _pipeline.cfg.dense_weight  = req.dense_weight
            _pipeline.cfg.sparse_weight = 1.0 - req.dense_weight

        docs = _pipeline.query(req.query)
        return QueryResponse(
            query=req.query,
            results=[DocResult(content=d.page_content, metadata=d.metadata) for d in docs]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", status_code=201)
def ingest(req: IngestRequest):
    doc = Document(page_content=req.content, metadata=req.metadata)
    _pipeline.add_documents([doc])
    return {"status": "ok", "total_docs": len(_pipeline.docs)}


@app.get("/health")
def health():
    return {"status": "up", "doc_count": len(_pipeline.docs)}


# ── Schemas ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:        str
    top_k:        Optional[int]   = None
    dense_weight: Optional[float] = None
    show_prompt:  bool = False     # optionally return the raw prompt for debugging

    @field_validator("query")
    def query_not_blank(cls, v):
        if not v or not v.strip():
            raise ValueError("query cannot be blank")
        return v.strip()

    @field_validator("top_k")
    def top_k_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("top_k must be >= 1")
        return v

    @field_validator("dense_weight")
    def weight_in_range(cls, v):
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("dense_weight must be between 0 and 1")
        return v


class DocResult(BaseModel):
    rank:     int
    content:  str
    metadata: dict


class QueryResponse(BaseModel):
    query:      str
    answer:     str                      # ← the generated answer from Gemini
    sources:    List[DocResult]          # ← the docs that grounded the answer
    latency_ms: Dict[str, float]         # ← per-stage breakdown
    prompt:     Optional[str] = None     # ← only returned if show_prompt=True


class IngestRequest(BaseModel):
    content:  str
    metadata: dict = {}

    @field_validator("content")
    def content_not_blank(cls, v):
        if not v or not v.strip():
            raise ValueError("content cannot be blank")
        return v.strip()


# ── Error handlers ─────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": e["loc"][-1], "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"errors": errors})

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Timing middleware ──────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    ms       = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(ms)
    logger.info(f"{request.method} {request.url.path} — {ms}ms")
    return response


# ── Endpoints ──────────────────────────────────────────────────

@app.post("/api/rag/query", response_model=QueryResponse)
def query(req: QueryRequest):

    # apply runtime overrides
    if req.top_k:
        _pipeline.cfg.top_k_rerank = req.top_k
    if req.dense_weight:
        _pipeline.cfg.dense_weight  = req.dense_weight
        _pipeline.cfg.sparse_weight = 1.0 - req.dense_weight

    result = _pipeline.query(req.query)

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            DocResult(
                rank=i + 1,
                content=doc.page_content,
                metadata=doc.metadata
            )
            for i, doc in enumerate(result.retrieved_docs)
        ],
        latency_ms=result.latency_ms,
        # only expose raw prompt if caller explicitly asks — useful during dev
        prompt=result.prompt if req.show_prompt else None
    )


@app.post("/api/rag/ingest", status_code=201)
def ingest(req: IngestRequest):
    doc = Document(page_content=req.content, metadata=req.metadata)
    _pipeline.add_documents([doc])
    return {"status": "ok", "total_docs": len(_pipeline.docs)}


@app.get("/api/rag/health")
def health():
    return {
        "status":    "up",
        "doc_count": len(_pipeline.docs),
        "config":    vars(_pipeline.cfg)
    }
