import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

# load the .env file
load_dotenv()

# create a new instance of the ChatOpenAI class
chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")

question = HumanMessage(content="What is the weather like in Wedge Island, Western Australia?")

response = chat_llm.invoke([
    instructions,
    question
])

print(response.content)