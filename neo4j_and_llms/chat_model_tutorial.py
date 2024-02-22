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

prompt = PromptTemplate(
    template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Question: {question}
""",
    input_variables=["question"],
)

chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

#question = HumanMessage(content="What is the weather like in Wedge Island, Western Australia?")

response = chat_chain.invoke({"question": "What is the weather like in Wedge Island, Western Australia?"})

print(response)