import os
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import YouTubeSearchTool
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# load the .env file
load_dotenv()

# get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create a language model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# create a prompt template
prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

# create a memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create a language model chain
chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# create a YouTube search tool
youtube = YouTubeSearchTool()

# create an openai embeddings object
embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# create the Neo4jVector object
movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url="bolt://44.201.38.129:7687",
    username="neo4j",
    password="journals-cupful-lot",
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)

# create RetrievalQA chain
plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True
)

def run_retriever(query):
    results = plot_retriever.invoke({"query":query})
    # format the results
    movies = '\n.'.join([doc.metadata["title"] + " - " + doc.page_content for doc in results["source_documents"]])
    return movies

# the tool is essentially a function that can be called by the agent
# it provides the agent with a way to interact with the LLMChain
# this allows context-dependent specialized behavior
tools = [
    # create a tool from a function
    # This tool instructs the LLM how to react to questions about movies
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True,
    ),
    # create a tool from a function
    # this tool allows the LLM to search for movie trailers on YouTube
    Tool.from_function(
        name="Movie Trailer Search",
        description="For when you need to find a movie trailer. The question will be a string. Return a string.",
        func=youtube.run,
        return_direct=True
    ),
    # create RetrievalQA tool
    Tool.from_function(
        name="Movie Plot Graph Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )
]


# create an agent from the react-chat model
# this is a model that has been trained to chat about movies
# it is available on the langchain hub
agent_prompt = hub.pull("hwchase17/react-chat")

# create an agent executor
# this is a class that allows the agent to be invoked
# it also provides the agent with access to the tools
# and memory
agent = create_react_agent(llm, tools, agent_prompt)

# create an agent executor
# this is a class that allows the agent to be invoked and runs the agent
# importantly, it also provides the agent with access to the tools and memory
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools,
                               memory=memory,
                               max_iterations=3, # the maximum number of iterations the agent can run
                               verbose=True, # print the agent's output
                               handle_parsing_errors=True) # handle parsing errors

# chat is now ready to be invoked
# the agent will respond to questions about movies
while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])