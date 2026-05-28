"""Cypher corpus metadata for source selection and execution."""

NEO4J_URI = "neo4j+s://demo.neo4jlabs.com:7687"

# Map text2cypher database_reference_alias → actual database name on demo.neo4jlabs.com.
ALIAS_TO_DB = {
    "neo4jlabs_demo_db_bluesky":         "bluesky",
    "neo4jlabs_demo_db_buzzoverflow":    "buzzoverflow",
    "neo4jlabs_demo_db_companies":       "companies",
    "neo4jlabs_demo_db_eoflix":          "neoflix",  # dataset typo: alias says eoflix but the hosted db is neoflix
    "neo4jlabs_demo_db_fincen":          "fincen",
    "neo4jlabs_demo_db_gameofthrones":   "gameofthrones",
    "neo4jlabs_demo_db_grandstack":      "grandstack",
    "neo4jlabs_demo_db_movies":          "movies",
    "neo4jlabs_demo_db_network":         "network",
    "neo4jlabs_demo_db_northwind":       "northwind",
    "neo4jlabs_demo_db_offshoreleaks":   "offshoreleaks",
    "neo4jlabs_demo_db_recommendations": "recommendations",
    "neo4jlabs_demo_db_stackoverflow2":  "stackoverflow2",
    "neo4jlabs_demo_db_twitch":          "twitch",
    "neo4jlabs_demo_db_twitter":         "twitter",
}
