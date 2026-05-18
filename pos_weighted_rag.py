import os
import re
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import spacy
import google.generativeai as genai
from langchain_core.documents import Document


@dataclass
class PipelineConfig:
    top_k_matches: int = 3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    enable_expansion: bool = True
    expansion_topk: int = 2

    gemini_model: str = "gemini-1.5-flash"
    max_output_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.85
    max_context_chars: int = 3000


def _tech_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s/_-]", " ", text)
    raw = text.split()
    expanded = []

    for tok in raw:
        sub = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok).split()
        expanded.append(tok.lower())

        if len(sub) > 1:
            expanded.extend(s.lower() for s in sub)

    return expanded


class HandRolledBM25:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.corpus = [_tech_tokenize(d.page_content) for d in docs]
        self.N = len(self.corpus)

        if self.N:
            self.avgdl = max(sum(len(d) for d in self.corpus) / self.N, 1.0)
        else:
            self.avgdl = 1.0

        self.df: Dict[str, int] = {}
        for tokens in self.corpus:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_weights: Dict[str, float], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        if not doc:
            return 0.0

        tf_map = Counter(doc)
        dl = len(doc)
        total = 0.0

        for term, weight in query_weights.items():
            if term not in tf_map:
                continue

            tf = tf_map[term]
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            bm25_term_score = idf * (tf * (self.k1 + 1)) / denom

            total += bm25_term_score * weight

        return total

    def retrieve(self, query_weights: Dict[str, float], k: int) -> List[Tuple[Document, float]]:
        if not self.docs or not query_weights:
            return []

        scores = [(i, self.score(query_weights, i)) for i in range(self.N)]
        scores = [(i, score) for i, score in scores if score > 0]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(self.docs[i], score) for i, score in scores[:k]]


nlp = spacy.load("en_core_web_sm")

_POS_SCORE_BOOST: Dict[str, float] = {
    "PROPN": 2.2,
    "NOUN": 1.6,
    "VERB": 1.1,
    "ADJ": 1.0,
}

_STOPWORD_POS = {"DET", "ADP", "PRON", "PART", "PUNCT", "SPACE"}

try:
    import nltk
    from nltk.corpus import wordnet

    _WN_AVAILABLE = True
except ImportError:
    _WN_AVAILABLE = False

_WORDNET_READY: Optional[bool] = None


def _ensure_wordnet_ready() -> bool:
    global _WORDNET_READY

    if not _WN_AVAILABLE:
        return False

    if _WORDNET_READY is not None:
        return _WORDNET_READY

    try:
        wordnet.ensure_loaded()
        _WORDNET_READY = True
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            wordnet.ensure_loaded()
            _WORDNET_READY = True
        except Exception:
            _WORDNET_READY = False

    return _WORDNET_READY


def expand_query(query: str, cfg: PipelineConfig) -> str:
    if not cfg.enable_expansion or not _ensure_wordnet_ready():
        return query

    doc = nlp(query)
    extras = []

    for token in doc:
        if token.pos_ != "NOUN":
            continue

        synsets = wordnet.synsets(token.lemma_)[: cfg.expansion_topk]

        for syn in synsets:
            for lemma in syn.lemmas()[:1]:
                word = lemma.name().replace("_", " ")

                if word.lower() != token.lower_:
                    extras.append(word)

    return query + " " + " ".join(extras) if extras else query


def pos_weighted_query_tokens(query: str) -> Dict[str, float]:
    doc = nlp(query)
    result: Dict[str, float] = {}

    for token in doc:
        if token.pos_ in _STOPWORD_POS or token.is_space:
            continue

        boost = _POS_SCORE_BOOST.get(token.pos_, 1.0)

        if token.ent_type_:
            boost *= 1.3

        for term in _tech_tokenize(token.text):
            result[term] = max(result.get(term, 1.0), boost)

    return result


class PromptBuilder:
    def __init__(self, max_context_chars: int = 4000):
        self.max_context_chars = max_context_chars

    def build(self, query: str, docs: List[Document]) -> str:
        context_string = ""

        for i, doc in enumerate(docs):
            source = doc.metadata.get(
                "chapter",
                doc.metadata.get("source", "Unknown Source"),
            )

            chunk_index = doc.metadata.get("chunk_index")
            if chunk_index is not None:
                source = f"{source}, chunk {chunk_index}"

            doc_entry = (
                f"--- Excerpt {i + 1} from {source} ---\n"
                f"{doc.page_content}\n\n"
            )

            if len(context_string) + len(doc_entry) > self.max_context_chars:
                break

            context_string += doc_entry

        return f"""You are WizGuide, a wise, friendly, and slightly scholarly magical librarian at Hogwarts.
A student has asked you a question.

Read the provided book excerpts carefully.
Use ONLY the information in the excerpts to answer the student's question.
If the answer is not contained in the excerpts, politely inform the student that you
have not found that information in the Restricted Section.
Do not make up facts outside of the provided text.

STUDENT QUESTION: {query}

BOOK EXCERPTS:
{context_string}

Answer the student now in a magical tone:"""


class GeminiGenerator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(cfg.gemini_model)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.cfg.max_output_tokens,
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                },
            )

            return response.text.strip()
        except Exception as e:
            return f"Alas, a magical disturbance blocked my vision! (Error: {str(e)})"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_docs_from_folder(folder: str) -> List[Document]:
    docs = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename},
                )
            )

    return docs


def load_and_chunk_docs(folder: str, chunk_size: int = 800) -> List[Document]:
    raw_docs = load_docs_from_folder(folder)
    chunked = []

    for doc in raw_docs:
        chunks = chunk_text(doc.page_content, chunk_size=chunk_size)

        for i, chunk in enumerate(chunks):
            chunked.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

    return chunked


@dataclass
class RAGResult:
    query: str
    retrieved_docs: List[Document]
    prompt: str
    answer: str
    latency_ms: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, docs: List[Document], cfg: PipelineConfig = PipelineConfig()):
        self.cfg = cfg
        self.docs = docs
        self.bm25 = HandRolledBM25(docs, k1=cfg.bm25_k1, b=cfg.bm25_b)
        self.prompt_builder = PromptBuilder(max_context_chars=cfg.max_context_chars)
        self.generator = GeminiGenerator(cfg)

    def add_documents(self, new_docs: List[Document]) -> None:
        self.docs.extend(new_docs)
        self.bm25 = HandRolledBM25(self.docs, k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)

    def query(self, raw_query: str) -> RAGResult:
        latency = {}

        t0 = time.perf_counter()
        expanded_query = expand_query(raw_query, self.cfg)
        latency["expansion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        token_weights = pos_weighted_query_tokens(expanded_query)
        latency["pos_weighting_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        matches = self.bm25.retrieve(token_weights, k=self.cfg.top_k_matches)
        docs = [doc for doc, _score in matches]
        latency["matching_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        prompt = self.prompt_builder.build(raw_query, docs)

        t0 = time.perf_counter()
        answer = self.generator.generate(prompt)
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return RAGResult(
            query=raw_query,
            retrieved_docs=docs,
            prompt=prompt,
            answer=answer,
            latency_ms=latency,
        )
