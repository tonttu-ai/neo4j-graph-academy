import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# load the .env file
load_dotenv()

# create a graph object, including connection to the neo4j database
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

result = graph.query("""
MATCH (m:Movie{title: 'Toy Story'}) 
RETURN m.title, m.plot, m.poster
""")

print(result)