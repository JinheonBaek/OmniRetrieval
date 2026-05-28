"""SPARQL corpus metadata for source selection."""

SPARQL_CORPORA: dict[str, dict[str, str | list[str]]] = {
    "wikidata": {
        "description": (
            "Structured knowledge graph queries about real-world "
            "entities (people, places, organizations, concepts), supporting "
            "property lookup, multi-hop traversal, aggregation, filtering, "
            "boolean verification, counting, ranking, and temporal constraints"
        ),
        "examples": [
            "who is the director of ladyhawke",                               # simplequestions
            "Who is the spouse of the person who painted Glorious victory?",  # lcquad2
            "Which movies starring Brad Pitt were directed by Guy Ritchie?",  # qald10
        ],
    },
}
