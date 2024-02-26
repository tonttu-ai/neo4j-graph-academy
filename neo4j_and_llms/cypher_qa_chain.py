import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# load the .env file
load_dotenv()

# get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create a language model object
llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature=1)

# create a graph object, including connection to the neo4j database
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

cypher_chain.invoke({"query": "What role did Tom Hanks play in Toy Story?"})
