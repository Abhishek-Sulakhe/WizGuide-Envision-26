# rag_engine.py

import re
import math
import spacy
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────
# CONFIG — all tunables live here, nowhere else
# ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    top_k_dense:       int   = 4
    top_k_sparse:      int   = 4
    top_k_rerank:      int   = 3
    dense_weight:      float = 0.35
    sparse_weight:     float = 0.65
    bm25_k1:           float = 1.5   # term saturation — lower = less saturation
    bm25_b:            float = 0.75  # doc length normalization
    enable_expansion:  bool  = True
    expansion_topk:    int   = 2     # how many synonyms to inject per PROPN
    gemini_model:      str   = "gemini-1.5-flash"
    max_output_tokens: int   = 1024
    temperature:       float = 0.2     # low = factual, deterministic answers
    top_p:             float = 0.85
    max_context_chars: int   = 3000    # hard cap on how much doc text goes into the prompt

# ──────────────────────────────────────────────
# CUSTOM BM25 — no langchain wrapper, full control
# ──────────────────────────────────────────────

def _tech_tokenize(text: str) -> List[str]:
    """
    Splits on whitespace + punctuation but keeps camelCase/underscore/slash
    tokens intact. 'REST_API' stays one token; 'MLForge' stays one token.
    """
    text = re.sub(r"[^\w\s/_-]", " ", text)
    raw = text.split()
    expanded = []
    for tok in raw:
        # also split camelCase so 'MLForge' → ['ML', 'Forge'] but keep original
        sub = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tok).split()
        expanded.append(tok.lower())
        if len(sub) > 1:
            expanded.extend(s.lower() for s in sub)
    return expanded


class HandRolledBM25:
    """
    Okapi BM25 written from scratch.
    Reason: langchain's BM25Retriever doesn't expose k1/b, 
    and rank_bm25 uses a generic tokenizer that mangles tech terms.
    """

    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs   = docs
        self.k1     = k1
        self.b      = b
        self.corpus = [_tech_tokenize(d.page_content) for d in docs]
        self.N      = len(self.corpus)
        self.avgdl  = sum(len(d) for d in self.corpus) / self.N

        # df[term] = number of docs containing term
        self.df: Dict[str, int] = {}
        for tokens in self.corpus:
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc    = self.corpus[doc_idx]
        tf_map = Counter(doc)
        dl     = len(doc)
        total  = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf    = tf_map[term]
            idf   = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * (tf * (self.k1 + 1)) / denom
        return total

    def retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        q_tokens = _tech_tokenize(query)
        scores   = [(i, self.score(q_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], s) for i, s in scores[:k]]


# ──────────────────────────────────────────────
# POS WEIGHT ENGINE — with soft-scoring instead
# of token repetition (the old approach was hacky)
# ──────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")

# Maps POS → score multiplier applied to BM25 hits per term
_POS_SCORE_BOOST: Dict[str, float] = {
    "PROPN": 2.2,   # product names, brands — highest signal
    "NOUN":  1.6,
    "VERB":  1.1,
    "ADJ":   1.0,
    # everything else → no boost (1.0 default below)
}

_STOPWORD_POS = {"DET", "ADP", "PRON", "PART", "PUNCT", "SPACE"}


def pos_weighted_query_tokens(query: str) -> Dict[str, float]:
    """
    Returns {token: boost_weight} instead of repeating tokens.
    Downstream BM25 uses this to post-multiply term scores.
    """
    doc    = nlp(query)
    result = {}
    for token in doc:
        if token.pos_ in _STOPWORD_POS or token.is_space:
            continue
        boost = _POS_SCORE_BOOST.get(token.pos_, 1.0)
        # named entities get an extra nudge
        if token.ent_type_:
            boost *= 1.3
        result[token.lower_] = boost
    return result


# ──────────────────────────────────────────────
# QUERY EXPANSION — injects WordNet synonyms for
# NOUNs only (expanding PROPNs causes drift)
# ──────────────────────────────────────────────

try:
    from nltk.corpus import wordnet
    import nltk
    nltk.download("wordnet", quiet=True)
    _WN_AVAILABLE = True
except ImportError:
    _WN_AVAILABLE = False


def expand_query(query: str, cfg: PipelineConfig) -> str:
    if not cfg.enable_expansion or not _WN_AVAILABLE:
        return query

    doc    = nlp(query)
    extras = []
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        synsets = wordnet.synsets(token.lemma_)[:cfg.expansion_topk]
        for syn in synsets:
            for lemma in syn.lemmas()[:1]:   # one lemma per synset
                word = lemma.name().replace("_", " ")
                if word.lower() != token.lower_:
                    extras.append(word)
    return query + " " + " ".join(extras) if extras else query


# ──────────────────────────────────────────────
# RECIPROCAL RANK FUSION — hand-rolled, no deps
# ──────────────────────────────────────────────

def _rrf(
    dense_hits:  List[Tuple[Document, float]],
    sparse_hits: List[Tuple[Document, float]],
    dense_w:     float,
    sparse_w:    float,
    k:           int = 60,
) -> List[Document]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def _key(doc: Document) -> str:
        return doc.page_content[:80]   # content fingerprint

    for rank, (doc, _) in enumerate(dense_hits):
        key = _key(doc)
        scores[key]  = scores.get(key, 0) + dense_w / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(sparse_hits):
        key = _key(doc)
        scores[key]  = scores.get(key, 0) + sparse_w / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked]

# ──────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs the final prompt that gets sent to Gemini.
    Keeps retrieved context, source labels, and the question
    clearly separated so the model doesn't hallucinate.
    """

    def __init__(self, max_context_chars: int = 3000):
        self.max_context_chars = max_context_chars

    def build(self, query: str, docs: List[Document]) -> str:
        context_blocks = []
        total_chars    = 0

        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get("doc_type", "general")
            snippet  = doc.page_content.strip()

            # enforce the context budget — stop adding docs once limit is hit
            if total_chars + len(snippet) > self.max_context_chars:
                snippet = snippet[: self.max_context_chars - total_chars]
                context_blocks.append(
                    f"[Source {i + 1} | type: {doc_type}]\n{snippet}"
                )
                break

            context_blocks.append(
                f"[Source {i + 1} | type: {doc_type}]\n{snippet}"
            )
            total_chars += len(snippet)

        context_text = "\n\n".join(context_blocks)

        # ── Prompt Template ───────────────────────────────────────
        # Three distinct sections: role, context, task.
        # "ONLY use the sources" prevents hallucination.
        # "cite [Source N]" makes answers traceable.

        prompt = f"""You are a precise technical assistant. \
Answer questions strictly based on the provided sources.

══ RETRIEVED CONTEXT ══
{context_text}

══ QUESTION ══
{query}

══ INSTRUCTIONS ══
- Answer concisely and factually using ONLY the sources above.
- If the sources do not contain enough information, say: \
"The available documents do not cover this clearly."
- Where possible, cite which [Source N] your answer comes from.
- Do not fabricate details not present in the sources.

══ ANSWER ══"""

        return prompt
    
    

# ──────────────────────────────────────────────
# GEMINI GENERATOR
# ──────────────────────────────────────────────

import google.generativeai as genai
import os

class GeminiGenerator:
    """
    Thin wrapper around the Gemini API.
    Handles initialization, config, and error cases
    without leaking SDK details into the pipeline.
    """

    def __init__(self, cfg: PipelineConfig):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Export it before starting the server."
            )

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=cfg.gemini_model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)

            # Gemini sometimes blocks responses on safety grounds —
            # surface that clearly instead of returning empty string
            if not response.candidates:
                return "Response blocked by safety filter."

            return response.text.strip()

        except Exception as e:
            # don't crash the whole request — return the error as the answer
            # so the retrieved docs still get returned to the caller
            return f"Generation failed: {str(e)}"
        
# ──────────────────────────────────────────────
# CROSS-ENCODER RERANKER
# ──────────────────────────────────────────────

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return docs
        pairs  = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Splits long text into overlapping windows.
    overlap ensures context isn't lost at chunk boundaries.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # step back by overlap amount

    return chunks

import os

# ==========================================
# LOAD DOCUMENTS FROM FOLDER
# ==========================================

def load_docs_from_folder(folder: str):

    docs = []

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        # Only load txt files
        if filename.endswith(".txt"):

            with open(path, "r", encoding="utf-8") as f:

                text = f.read()

                docs.append(

                    Document(

                        page_content=text,

                        metadata={
                            "source": filename
                        }
                    )
                )

    return docs



def load_and_chunk_docs(folder: str, chunk_size: int = 800) -> List[Document]:
    raw_docs = load_docs_from_folder(folder)
    chunked  = []

    for doc in raw_docs:
        chunks = chunk_text(doc.page_content, chunk_size=chunk_size)
        for i, chunk in enumerate(chunks):
            chunked.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,        # which chunk of the original doc
                    "total_chunks": len(chunks)
                }
            ))

    return chunked


# ──────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, docs: List[Document], cfg: PipelineConfig = PipelineConfig()):
        self.cfg      = cfg
        self.docs     = docs
        self.bm25     = HandRolledBM25(docs, k1=cfg.bm25_k1, b=cfg.bm25_b)
        self.reranker = Reranker()

        emb             = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(docs, emb)

    def add_documents(self, new_docs: List[Document]):
        """Hot-add docs without rebuilding everything."""
        self.docs.extend(new_docs)
        self.bm25 = HandRolledBM25(self.docs, k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        self.vectorstore.add_documents(new_docs)

    def query(self, raw_query: str) -> List[Document]:
        cfg = self.cfg
        # 1. Expand
        expanded = expand_query(raw_query, cfg)

        # 2. POS token weights
        token_boosts = pos_weighted_query_tokens(expanded)

        # 3. Dense retrieval
        dense_raw   = self.vectorstore.similarity_search_with_score(expanded, k=cfg.top_k_dense)
        dense_hits  = [(doc, score) for doc, score in dense_raw]

        # 4. Sparse retrieval + POS post-multiplication
        sparse_raw  = self.bm25.retrieve(expanded, k=cfg.top_k_sparse)
        sparse_hits = []
        for doc, base_score in sparse_raw:
            boost = 1.0
            for tok, w in token_boosts.items():
                if tok in doc.page_content.lower():
                    boost = max(boost, w)
            sparse_hits.append((doc, base_score * boost))
        sparse_hits.sort(key=lambda x: x[1], reverse=True)

        # 5. Fuse
        fused = _rrf(dense_hits, sparse_hits, cfg.dense_weight, cfg.sparse_weight)

        # 6. Rerank
        return self.reranker.rerank(raw_query, fused, top_k=cfg.top_k_rerank)