"""
OmniRetrieval pipeline: source selection, query formulation, execution, evidence selection.

Source selection:   question → up to top_k RouteDecisions (route_type + kb_id)
Query formulation:  RouteDecision + structural context → GeneratedQuery
Execution:          GeneratedQuery → ExecutionResult (run against the source)
Evidence selection: candidates → index of the result that best answers the question
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.data.beir_corpora import BEIR_CORPORA, extract_corpus_sentences
from src.data.sparql_corpora import SPARQL_CORPORA
from src.data.schema import RouteType
from src.model.llm_client import LLMClient
from src.utils import execute_sql, execute_sparql, execute_cypher, hydrate_wikidata_qids


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

BASELINE_SEARCH_PROMPT = (
    "You are a search query optimizer. "
    "Output only the refined search query."
)

HYDE_SEARCH_PROMPT = (
    "You are a search query optimizer for a dense retriever. "
    "Given the user query and a description of the target corpus, write a "
    "hypothetical passage that would be relevant evidence for the query, written "
    "in the register, style, and approximate length of documents in that corpus. "
    "The passage will be embedded and matched against real corpus documents, so "
    "favor concrete, in-domain content over generic phrasing. "
    "Begin your output with the user query verbatim on its own line — just the "
    "query text, not the 'Question:' label that precedes it — then on the next "
    "line write the hypothetical passage. This keeps the literal query terms in "
    "the embedding alongside the semantic expansion. "
    "If the query is a short topic stub or short keyword-style query (just "
    "a handful of words, often lowercase and without punctuation), do NOT write a "
    "passage — output the verbatim query and nothing else. The bare term already "
    "gives a strong dense-retrieval signal, and a hallucinated passage tends to lock "
    "onto one specific aspect that may not match the gold document. "
    "Output only the verbatim query followed by the passage (or for a stub query, "
    "only the verbatim query) — no preamble, no quotes, no labels."
)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class RouteDecision:
    question: str
    route: RouteType
    kb_id: str

    def to_dict(self) -> dict:
        return {"question": self.question, "route": self.route.value, "kb_id": self.kb_id}

    @classmethod
    def from_dict(cls, d: dict) -> "RouteDecision":
        return cls(question=d["question"], route=RouteType(d["route"]), kb_id=d["kb_id"])


@dataclass
class GeneratedQuery:
    route_decision: RouteDecision
    query: str

    def to_dict(self) -> dict:
        return {"route_decision": self.route_decision.to_dict(), "query": self.query}

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedQuery":
        return cls(route_decision=RouteDecision.from_dict(d["route_decision"]), query=d["query"])


@dataclass
class ExecutionResult:
    generated_query: GeneratedQuery
    answer: list[str]                                       # SEARCH: top-k doc IDs | SQL: result values | SPARQL: entity URIs | CYPHER: result rows
    scores: dict[str, float] = field(default_factory=dict)  # SEARCH only: doc_id -> relevance score (for IR metrics)

    def to_dict(self) -> dict:
        return {
            "generated_query": self.generated_query.to_dict(),
            "answer": self.answer,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionResult":
        return cls(
            generated_query=GeneratedQuery.from_dict(d["generated_query"]),
            answer=d.get("answer", []),
            scores=d.get("scores", {}),
        )


@dataclass
class CandidateResults:
    """All execution results for one sample (multi-candidate source selection)."""
    candidates: list[ExecutionResult]
    selected: int = 0

    @property
    def best(self) -> ExecutionResult:
        return self.candidates[self.selected]

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "selected": self.selected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateResults":
        return cls(
            candidates=[ExecutionResult.from_dict(c) for c in d["candidates"]],
            selected=d.get("selected", 0),
        )


# ------------------------------------------------------------------
# Retrieval pipeline
# ------------------------------------------------------------------


class RetrievalPipeline:
    def __init__(
        self,
        llm: LLMClient,
        data_dir: str = "data/processed",
        search_model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
    ):
        self.llm = llm
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self._kb_registry = self._load_kb_registry()
        self._corpus_cache: dict[str, tuple[list[str], torch.Tensor]] = {}  # kb_id -> (doc_ids, embeddings)
        self._corpus_text_cache: dict[str, dict[str, dict[str, str]]] = {}  # kb_id -> {doc_id: {title, text}}
        self._cache_lock = threading.RLock()

        from sentence_transformers import SentenceTransformer
        self.search_model_name = search_model
        self.search_model = SentenceTransformer(search_model)

    # ------------------------------------------------------------------
    # KB registry
    # ------------------------------------------------------------------

    def _load_kb_registry(self) -> dict[str, list[str]]:
        """Load available kb_ids for each backend from processed data."""
        registry: dict[str, list[str]] = {}

        # SEARCH: BEIR corpus names with descriptions and example queries
        beir_dir = self.data_dir / "beir"
        assert beir_dir.exists(), f"BEIR directory not found: {beir_dir}"
        search_items: list[str] = []
        for p in sorted(beir_dir.glob("*.jsonl"), key=lambda p: p.stem):
            meta = BEIR_CORPORA[p.stem]
            desc = meta["description"]
            qtype = meta["query_type"]
            exs = ", ".join(f'"{e}"' for e in meta["examples"])
            search_items.append(f"{p.stem} [{desc} | query type: {qtype} | examples: {exs}]")
        registry["SEARCH"] = search_items

        # SQL: database IDs from Spider + BIRD, with table names as summary
        sql_items: list[str] = []
        for source in ("spider", "bird"):
            db_dir = self.data_dir / source / "databases"
            assert db_dir.exists(), f"Database directory not found: {db_dir}"
            for folder in sorted(db_dir.iterdir()):
                if not folder.is_dir():
                    continue
                db_path = folder / f"{folder.name}.sqlite"
                if db_path.exists():
                    tables = self._extract_tables_only(db_path)
                    sql_items.append(f"{folder.name} [{tables}]")
        registry["SQL"] = sql_items

        # SPARQL: knowledge graphs
        sparql_items: list[str] = []
        for kb_id, meta in SPARQL_CORPORA.items():
            desc = meta["description"]
            exs = ", ".join(f'"{e}"' for e in meta["examples"])
            sparql_items.append(f"{kb_id} [{desc} | examples: {exs}]")
        registry["SPARQL"] = sparql_items

        # CYPHER: Neo4j databases from text2cypher, with node labels + relationships as summary
        cypher_dir = self.data_dir / "text2cypher" / "databases"
        assert cypher_dir.exists(), f"Cypher database directory not found: {cypher_dir}"
        cypher_items: list[str] = []
        for folder in sorted(cypher_dir.iterdir()):
            if not folder.is_dir():
                continue
            schema_path = folder / "schema.txt"
            if schema_path.exists():
                summary = self._extract_neo4j_summary(schema_path.read_text(encoding="utf-8"))
                cypher_items.append(f"{folder.name} [{summary}]")
        registry["CYPHER"] = cypher_items

        return registry

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def _load_context(self, route_type: RouteType, kb_id: str, question: str = "", metadata: dict | None = None) -> str:
        """Load the structural context for query formulation."""
        if route_type == RouteType.SEARCH:
            return self._load_search_context(kb_id)
        elif route_type == RouteType.SQL:
            return self._load_sql_schema(kb_id)
        elif route_type == RouteType.SPARQL:
            return self._load_sparql_context(kb_id, question, metadata)
        elif route_type == RouteType.CYPHER:
            return self._load_cypher_context(kb_id)
        raise ValueError(f"Unsupported route type: {route_type}")

    def _load_search_context(self, kb_id: str) -> str:
        metadata = BEIR_CORPORA.get(kb_id)
        if metadata is None:
            return f"Corpus: {kb_id}"
        return (
            f"Corpus: {kb_id}\n"
            f"Description: {metadata['description']}\n"
            f"Query type: {metadata['query_type']}\n"
            f"Document style: {metadata['doc_style']}"
        )

    def _find_db_path(self, kb_id: str) -> Path:
        """Find the SQLite database path for a given kb_id."""
        for source in ("spider", "bird"):
            db_path = self.data_dir / source / "databases" / kb_id / f"{kb_id}.sqlite"
            if db_path.exists():
                return db_path
        raise FileNotFoundError(f"Database not found for '{kb_id}'")

    def _load_sql_schema(self, kb_id: str) -> str:
        """Extract schema from the SQLite database file."""
        try:
            return self._extract_schema(self._find_db_path(kb_id))
        except FileNotFoundError:
            return f"Database: {kb_id} (schema not available)"

    def _extract_tables_only(self, db_path: Path) -> str:
        """Return comma-separated table names from a SQLite database."""
        conn = sqlite3.connect(str(db_path))
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            tables = [t for (t,) in tables if t != "sqlite_sequence"]
        finally:
            conn.close()
        return ", ".join(tables)

    def _extract_schema(self, db_path: Path) -> str:
        """Return full CREATE TABLE DDL statements from a SQLite database."""
        conn = sqlite3.connect(str(db_path))
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            tables = [t for (t,) in tables if t != "sqlite_sequence"]
            schemas = {}
            for table in tables:
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                schemas[table] = row[0]
        finally:
            conn.close()
        return "Database schema:\n\n" + "\n\n".join(
            schemas[t] for t in sorted(schemas)
        )

    def _extract_neo4j_summary(self, schema_text: str) -> str:
        """Return 'nodes: A, B, ... | rels: (X)-[R]->(Y), ...' from a Neo4j schema text."""
        node_block = re.search(
            r"Node properties:(.*?)(?:Relationship properties:|The relationships:|\Z)",
            schema_text, re.S,
        )
        labels = (
            re.findall(r"-\s*\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*", node_block.group(1))
            if node_block else []
        )
        if not labels:
            labels = re.findall(r"^([A-Z][A-Za-z0-9_]*)\s*\{", schema_text, re.M)
        rels = re.findall(r"\(:([A-Za-z_]+)\)-\[:([A-Za-z_]+)\]->\(:([A-Za-z_]+)\)", schema_text)
        return (
            f"nodes: {', '.join(labels)} | "
            f"rels: {', '.join(f'({h})-[{r}]->({t})' for h, r, t in rels)}"
        )

    def _rank_relations(self, question: str, relations: dict[str, str], top_k: int) -> dict[str, str]:
        """Return the top-k properties whose labels are most semantically similar to the question."""
        if not question or len(relations) <= top_k:
            return relations
        pids, labels = list(relations.keys()), list(relations.values())
        q_emb = self.search_model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        l_emb = self.search_model.encode(labels, normalize_embeddings=True, convert_to_numpy=True)
        scores = (q_emb @ l_emb.T)[0]
        top = scores.argsort()[-top_k:][::-1]
        return {pids[i]: labels[i] for i in top}

    def _load_sparql_context(self, kb_id: str, question: str = "", metadata: dict | None = None, top_k: int = 30) -> str:
        base = (
            "Knowledge graph: Wikidata\n"
            "Prefixes: wd: (entity), wdt: (direct/truthy property), "
            "p: (entity → statement node), ps: (statement → main value), "
            "pq: (statement → qualifier value), rdfs: (for rdfs:label)\n\n"
            "Format examples (schematic IDs — use the actual topic entities and relations for the query):\n"
            "  SELECT ?x WHERE { wd:Qxxx wdt:Pyyy ?x }\n"
            "  SELECT ?x WHERE { wd:Qxxx wdt:Pyyy ?y . ?y wdt:Pzzz ?x }\n"
            "  ASK WHERE { wd:Qxxx wdt:Pyyy wd:Qzzz }\n"
            "  SELECT ?v ?q WHERE { wd:Qxxx p:Pyyy ?s . ?s ps:Pyyy ?v . ?s pq:Pqqq ?q }\n"
            "  SELECT ?x ?l WHERE { ?x wdt:Pyyy wd:Qxxx . ?x rdfs:label ?l . FILTER(CONTAINS(lcase(?l), \"kw\")) . FILTER(lang(?l) = \"en\") }\n"
            "  SELECT (COUNT(?x) AS ?n) WHERE { wd:Qxxx wdt:Pyyy ?x }\n"
            "  SELECT ?x WHERE { wd:Qxxx wdt:Pyyy ?x . ?x wdt:Pzzz ?d } ORDER BY ?d LIMIT 1"
        )
        if not metadata:
            return base

        topic_entities = metadata.get("topic_entities") or {}
        if not topic_entities:
            return base

        entity_relations = {
            qid: self._rank_relations(question, pids, top_k)
            for qid, pids in (metadata.get("entity_relations") or {}).items()
        }

        entities_str = "\n".join(
            f"- wd:{qid} ({label})" for qid, label in topic_entities.items()
        )
        relations_str = "\n".join(
            f"- wd:{qid} ({topic_entities.get(qid, '')}):\n"
            + "\n".join(f"    - {pid} ({plabel})" for pid, plabel in pids.items())
            for qid, pids in entity_relations.items() if pids
        )

        return (
            f"{base}\n\n"
            f"Topic entities (from the question, already linked to Wikidata QIDs):\n{entities_str}\n\n"
            f"Linked relations (candidate properties per topic entity — choose among these):\n{relations_str}"
        )

    def _load_cypher_context(self, kb_id: str) -> str:
        schema_path = self.data_dir / "text2cypher" / "databases" / kb_id / "schema.txt"
        return schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""

    # ------------------------------------------------------------------
    # Source selection: backend + knowledge base
    # ------------------------------------------------------------------

    def route(
        self,
        question: str,
        top_k: int = 1,
        single_source: RouteType | None = None,
    ) -> list[RouteDecision]:
        """
        Source selection: predict which backend and knowledge base.

        Returns up to top_k RouteDecisions (most likely first).
        """
        system = (
            "You are a query router. Given a question, decide which backend "
            "to use (SEARCH, SQL, SPARQL, or CYPHER) and which knowledge base to query. "
            "Some queries are ambiguous and may match multiple knowledge bases — "
            f"return up to {top_k} routing decisions, most likely first; "
            "return fewer if you are confident."
        )

        registry = (
            {single_source.value: self._kb_registry[single_source.value]}
            if single_source else self._kb_registry
        )

        kb_list = "\n\n".join(
            f"  {rtype}:\n" + "\n".join(f"    - {kb_item}" for kb_item in kb_items)
            for rtype, kb_items in registry.items()
        )

        prompt = (
            f"Available knowledge bases:\n\n{kb_list}\n\n"
            f"Question: {question}\n\n"
            'Respond with JSON: {"decisions": [{"route_type": "...", "kb_id": "..."}, ...]}'
        )

        result = self.llm.generate_json(prompt, system=system)

        decisions = [
            RouteDecision(
                question=question,
                route=RouteType(d["route_type"]) if not single_source else single_source,
                kb_id=d["kb_id"],
            )
            for d in result["decisions"][:top_k]
        ]
        if not decisions:
            raise ValueError(f"Router returned no decisions for question: {question!r}")
        return decisions

    # ------------------------------------------------------------------
    # Query formulation
    # ------------------------------------------------------------------

    def generate(self, decision: RouteDecision, metadata: dict | None = None) -> GeneratedQuery:
        """
        Query formulation: given the selected source,
        look up its structural context and formulate the native query.

        metadata is the sample-level metadata (e.g. SPARQL topic_entities and
        entity_relations) used to build a per-sample context.
        """
        context = self._load_context(decision.route, decision.kb_id, decision.question, metadata)

        system_by_type = {
            RouteType.SEARCH: HYDE_SEARCH_PROMPT if metadata else BASELINE_SEARCH_PROMPT,
            RouteType.SQL: (
                "You are a text-to-SQL translator. "
                "Output only the SQL query."
            ),
            RouteType.SPARQL: (
                "You are a text-to-SPARQL translator. "
                "Output only the SPARQL query."
            ),
            RouteType.CYPHER: (
                "You are a text-to-Cypher translator. "
                "Output only the Cypher query."
            ),
        }

        prompt = (
            f"{context}\n\n"
            f"Question: {decision.question}\n\n"
            f"Generate the {decision.route.value} query."
        )

        query = self.llm.generate(
            prompt, system=system_by_type[decision.route]
        ).strip()
        # Strip ```sql|cypher|sparql ... ``` fences some models wrap code in; no-op otherwise.
        query = re.sub(r"^\s*```(?:sql|cypher|sparql)\s*(.*?)\s*```\s*$", r"\1", query, flags=re.DOTALL | re.IGNORECASE).strip()

        return GeneratedQuery(route_decision=decision, query=query)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _load_corpus(self, kb_id: str, split: str = "test") -> dict[str, dict[str, str]]:
        """Load BEIR corpus from JSONL into {doc_id: {title, text}} format."""
        with self._cache_lock:
            if kb_id not in self._corpus_text_cache:
                self._corpus_text_cache[kb_id] = self._read_corpus_file(kb_id, split)
            return self._corpus_text_cache[kb_id]

    def _read_corpus_file(self, kb_id: str, split: str = "test") -> dict[str, dict[str, str]]:
        corpus_dir = self.data_dir / "beir" / "corpus"
        path = corpus_dir / f"{kb_id}_corpus.jsonl"
        if not path.exists():
            path = corpus_dir / f"{kb_id}_{split}_corpus.jsonl"
        if not path.exists():
            print(f"Skipping corpus load for unknown kb '{kb_id}'")
            return {}

        corpus = {}
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc.get("text", "")}
        return corpus

    def _get_beir_cache_dir(self) -> Path:
        """Return the disk cache directory for BEIR corpus embeddings."""
        model_name = self.search_model_name.replace("/", "_")
        return self.data_dir.parent / "cache" / "beir" / model_name

    def _get_corpus_index(self, kb_id: str, split: str = "test") -> tuple[list[str], torch.Tensor]:
        """Load corpus embeddings from disk cache, or encode and save."""
        # Use kb_id as cache name if a single corpus exists, otherwise append split
        corpus_dir = self.data_dir / "beir" / "corpus"
        cache_name = kb_id if (corpus_dir / f"{kb_id}_corpus.jsonl").exists() else f"{kb_id}_{split}"

        with self._cache_lock:
            if cache_name not in self._corpus_cache:
                self._corpus_cache[cache_name] = self._build_corpus_index(kb_id, cache_name, split)
            return self._corpus_cache[cache_name]

    def _build_corpus_index(self, kb_id: str, cache_name: str, split: str) -> tuple[list[str], torch.Tensor]:
        cache_dir = self._get_beir_cache_dir()
        doc_ids_path = cache_dir / f"{cache_name}_doc_ids.json"
        embeddings_path = cache_dir / f"{cache_name}_embeddings.pt"

        if doc_ids_path.exists() and embeddings_path.exists():
            with open(doc_ids_path) as f:
                doc_ids = json.load(f)
            embeddings = torch.load(embeddings_path, weights_only=True)
            print(f"Loaded cached embeddings for '{cache_name}' ({len(doc_ids)} docs)")
        else:
            corpus = self._load_corpus(kb_id, split)
            doc_ids = list(corpus.keys())
            texts = extract_corpus_sentences(
                [corpus[d] for d in doc_ids], sep=" ",
            )
            print(f"Encoding {len(doc_ids)} documents for '{cache_name}'...")
            embeddings = self.search_model.encode(
                texts, batch_size=128, show_progress_bar=True,
                normalize_embeddings=True, convert_to_numpy=True,
            )
            embeddings = torch.from_numpy(embeddings)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(doc_ids_path, "w") as f:
                json.dump(doc_ids, f)
            torch.save(embeddings, embeddings_path)

        return doc_ids, embeddings

    def execute(self, generated: GeneratedQuery, top_k: int | None = None, split: str = "test") -> ExecutionResult:
        """
        Execution: run the native query against the target source
        and return the results.
        """
        route = generated.route_decision.route
        k = top_k or self.top_k
        if route == RouteType.SEARCH:
            return self._execute_search(generated, k, split)
        elif route == RouteType.SQL:
            return self._execute_sql(generated)
        elif route == RouteType.SPARQL:
            return self._execute_sparql(generated)
        elif route == RouteType.CYPHER:
            return self._execute_cypher(generated)
        raise ValueError(f"Unsupported route type: {route}")

    def _execute_search(self, generated: GeneratedQuery, top_k: int, split: str) -> ExecutionResult:
        """Run dense retrieval over a BEIR corpus using the generated query."""
        kb_id = generated.route_decision.kb_id
        query = generated.query

        corpus_dir = self.data_dir / "beir" / "corpus"
        if not (corpus_dir / f"{kb_id}_corpus.jsonl").exists() and \
           not (corpus_dir / f"{kb_id}_{split}_corpus.jsonl").exists():
            print(f"Skipping search for unknown kb '{kb_id}'")
            return ExecutionResult(generated_query=generated, answer=[], scores={})

        doc_ids, corpus_embeddings = self._get_corpus_index(kb_id, split)

        from beir.retrieval.search.dense.util import cos_sim

        query_embedding = self.search_model.encode([query], convert_to_numpy=True)
        query_embedding = torch.from_numpy(query_embedding)
        scores = cos_sim(query_embedding, corpus_embeddings)[0]

        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)), largest=True, sorted=True)
        result_scores = {doc_ids[i]: float(s) for i, s in zip(top_indices.tolist(), top_scores.tolist())}
        result_ids = list(result_scores.keys())

        return ExecutionResult(
            generated_query=generated,
            answer=result_ids,
            scores=result_scores,
        )

    def _execute_sql(self, generated: GeneratedQuery) -> ExecutionResult:
        """Execute a SQL query against the target SQLite database."""
        kb_id = generated.route_decision.kb_id

        if not any((self.data_dir / source / "databases" / kb_id / f"{kb_id}.sqlite").exists()
                   for source in ("spider", "bird")):
            print(f"Skipping SQL for unknown kb '{kb_id}'")
            return ExecutionResult(generated_query=generated, answer=[])

        db_path = self._find_db_path(kb_id)
        answer = execute_sql(generated.query, str(db_path))
        return ExecutionResult(generated_query=generated, answer=answer)

    def _execute_sparql(self, generated: GeneratedQuery) -> ExecutionResult:
        """Execute a SPARQL query against the Wikidata endpoint."""
        answer = execute_sparql(generated.query)
        return ExecutionResult(generated_query=generated, answer=answer)

    def _execute_cypher(self, generated: GeneratedQuery) -> ExecutionResult:
        """Execute a Cypher query against the target Neo4j Labs demo DB."""
        answer = execute_cypher(generated.query, generated.route_decision.kb_id)
        return ExecutionResult(generated_query=generated, answer=answer)

    # ------------------------------------------------------------------
    # Cross-source evidence selection
    # ------------------------------------------------------------------

    def _select_context_search(self, candidate: ExecutionResult) -> str:
        """Search context for evidence selection: title + text preview of top docs."""
        corpus = self._load_corpus(candidate.generated_query.route_decision.kb_id)
        return "results:\n" + "\n".join(
            f"  - [{doc_id}] {corpus.get(doc_id, {}).get('title', '')}: "
            f"{(corpus.get(doc_id, {}).get('text', '') or '')[:200]}"
            for doc_id in candidate.answer[:5]
        )

    def _select_context_sql(self, candidate: ExecutionResult) -> str:
        """SQL context for evidence selection: target database schema and result rows."""
        schema = self._load_sql_schema(candidate.generated_query.route_decision.kb_id)
        return f"\n{schema}\n\nresults: {candidate.answer[:100]}"

    def _select_context_sparql(self, candidate: ExecutionResult, metadata: dict | None, top_k: int = 30) -> str:
        """SPARQL context for evidence selection: linked topic entities + question-relevant relations and result values."""
        entities = (metadata or {}).get("topic_entities") or {}
        all_relations = (metadata or {}).get("entity_relations") or {}
        question = candidate.generated_query.route_decision.question
        relations = {
            qid: self._rank_relations(question, pids, top_k)
            for qid, pids in all_relations.items()
        }
        entity_lines = "\n".join(
            f"  wd:{qid} ({entity_label}): "
            + ", ".join(f"{pid} ({prop_label})" for pid, prop_label in (relations.get(qid) or {}).items())
            for qid, entity_label in entities.items()
        )
        results = hydrate_wikidata_qids(candidate.answer[:100])
        return f"entities/relations:\n{entity_lines}\nresults: {results}"

    def _select_context_cypher(self, candidate: ExecutionResult) -> str:
        """Cypher context for evidence selection: target database schema and result rows."""
        schema = self._load_cypher_context(candidate.generated_query.route_decision.kb_id)
        return f"\n{schema}\n\nresults: {candidate.answer[:100]}"

    def select_context(self, candidate: ExecutionResult, metadata: dict | None) -> str:
        """Render the structural context (schema + materialized answer) for one candidate."""
        route = candidate.generated_query.route_decision.route
        if route == RouteType.SEARCH:
            return self._select_context_search(candidate)
        elif route == RouteType.SQL:
            return self._select_context_sql(candidate)
        elif route == RouteType.SPARQL:
            return self._select_context_sparql(candidate, metadata)
        elif route == RouteType.CYPHER:
            return self._select_context_cypher(candidate)
        raise ValueError(f"Unsupported route type: {route}")

    def select(
        self,
        question: str,
        candidates: list[ExecutionResult],
        metadata: dict | None = None,
    ) -> int:
        """Pick the candidate index whose result best answers the question."""
        if len(candidates) <= 1:
            return 0

        blocks = []
        for index, candidate in enumerate(candidates):
            decision = candidate.generated_query.route_decision
            # Document Search queries can include the hallucinated hypothetical passage; use the literal question instead.
            blocks.append(
                f"[{index}] {decision.route.value} | {decision.kb_id}\n"
                f"query: {decision.question if decision.route == RouteType.SEARCH else candidate.generated_query.query}\n"
                f"{self.select_context(candidate, metadata)}"
            )

        system = "You are a result selector. Pick the candidate whose result best answers the question."
        prompt = (
            f"Question: {question}\n\n"
            f"Candidates (each prefixed with its integer index in brackets, e.g. [0], [1], [2]):\n\n"
            + "\n\n".join(blocks)
            + '\n\nRespond with JSON: {"selected": <integer index>}'
        )
        result = self.llm.generate_json(prompt, system=system)
        selected = result.get("selected")
        return (
            selected
            if isinstance(selected, int) and 0 <= selected < len(candidates)
            else 0
        )
