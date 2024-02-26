from langchain_community.graphs import Neo4jGraph

# create a graph object, including connection to the neo4j database
graph = Neo4jGraph(
    url="bolt://44.201.38.129:7687",
    username="neo4j",
    password="journals-cupful-lot"
)

result = graph.query("""
MATCH (m:Movie{title: 'Toy Story'}) 
RETURN m.title, m.plot, m.poster
""")

print(result)