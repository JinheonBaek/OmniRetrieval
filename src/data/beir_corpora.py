"""BEIR corpus metadata for source selection."""

BEIR_CORPORA: dict[str, dict[str, str | list[str]]] = {
    "fever": {
        "description": "Fact verification claims against Wikipedia passages",
        "query_type": (
            "a factual claim phrased as a single declarative sentence; "
            "may be either true (needing supporting evidence) or false (needing refuting evidence)"
        ),
        "doc_style": (
            "a brief Wikipedia passage with a title — typically only a sentence or two of "
            "encyclopedic prose stating facts about an entity, event, or year"
        ),
        "examples": [
            "Gone with the Wind was award-winning.",
            "The Divergent Series is an adaptation.",
        ],
    },
    "fiqa": {
        "description": "Financial opinion and question answering from web forums",
        "query_type": (
            "a personal-finance, taxes, or investing query — typically a question, "
            "but sometimes a short noun-phrase topic"
        ),
        "doc_style": (
            "an untitled forum reply on a finance topic — conversational and first-person, "
            "frequently giving opinion, anecdote, or practical advice rather than formal exposition"
        ),
        "examples": [
            "Can a company stop paying dividends?",
            "What ETF best tracks the price of gasoline, or else crude oil?",
        ],
    },
    "hotpotqa": {
        "description": "Multi-hop reasoning questions over Wikipedia passages",
        "query_type": (
            "a multi-hop natural-language question that joins two or more entities or facts; "
            "formal, properly cased, with punctuation"
        ),
        "doc_style": (
            "a titled Wikipedia paragraph that introduces or describes a single entity; "
            "a relevant document typically covers ONE hop of the question, not the entire chain, "
            "so the same query may need to retrieve several different documents"
        ),
        "examples": [
            "Which band has had more members, Ubiquitous Synergy Seeker or Super Furry Animals?",
            "The film Salome's Last Dance is based on a play by a writer that died in what year?",
        ],
    },
    "msmarco": {
        "description": "Short informal Bing queries on diverse everyday topics",
        "query_type": (
            "a short web query, almost always lowercase and without punctuation; "
            "usually a wh-question or a noun-phrase lookup ('what is X', 'population of Y', 'cost of Z')"
        ),
        "doc_style": (
            "an untitled web passage that addresses the query directly — "
            "typically a short, factual, paragraph-form snippet"
        ),
        "examples": [
            "what type of pork ribs",
            "population of douala",
        ],
    },
    "nfcorpus": {
        "description": "Medical and nutritional information retrieval",
        "query_type": (
            "a consumer-health or nutrition query, often very terse — either a one-to-three-word "
            "topic stub (a food, condition, or substance) or a popular-science headline phrased "
            "for a lay audience"
        ),
        "doc_style": (
            "a PubMed-style biomedical research abstract with a formal scientific title, "
            "written in technical register"
        ),
        "examples": [
            "kidney health",
            "Does Cranberry Juice Work Against Bladder Infections?",
        ],
    },
    "nq": {
        "description": "Google search queries with Wikipedia article answers",
        "query_type": (
            "a natural-language factual question, all lowercase and without punctuation, "
            "almost always starting with a wh-word"
        ),
        "doc_style": (
            "a titled Wikipedia paragraph in narrative prose that directly contains the answer "
            "to a factual question"
        ),
        "examples": [
            "who played b.a. baracus in the a-team",
            "who was the leader of italy during the second world war",
        ],
    },
    "scifact": {
        "description": "Scientific claim verification against research abstracts",
        "query_type": (
            "a single-sentence scientific claim phrased in technical, formal language; "
            "may be either true (needing supporting evidence) or false (needing refuting evidence)"
        ),
        "doc_style": (
            "a scientific research abstract with a title, written in formal scientific register"
        ),
        "examples": [
            "N348I mutations cause resistance to nevirapine.",
            "Sympathetic nerve activity is reduced throughout normal pregnancy.",
        ],
    },
}


def extract_corpus_sentences(corpus: list[dict[str, str]], sep: str) -> list[str]:
    return [
        (doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
        for doc in corpus
    ]
