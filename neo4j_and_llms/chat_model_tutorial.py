import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

# load the .env file
load_dotenv()

# create a new instance of the ChatOpenAI class
chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# create a new instance of the PromptTemplate class
prompt = PromptTemplate(
    template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)

# create a new instance of the LLMChain class
chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

# providing context
current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"},
            {"beach": "Wedge Island", "conditions": "4ft waves and offshore winds"}
        ]
    }"""

# invoke the chat_chain

response = chat_chain.invoke(
    {   
        "context": current_weather,
        "question": "Where is the best place to surf today?"
    }
)

print(response)