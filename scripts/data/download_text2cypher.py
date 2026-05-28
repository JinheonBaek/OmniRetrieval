"""Download Neo4j text2cypher dataset into data/raw/text2cypher/."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="neo4j/text2cypher-2024v1",
    repo_type="dataset",
    local_dir="data/raw/text2cypher",
)
