import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# load the .env file
load_dotenv()

# create a new instance of the ChatOpenAI class
chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# create a new instance of the PromptTemplate class
prompt = PromptTemplate(template="""
                                    You are a surfer dude, having a conversation about the surf conditions on the beach.
                                    Respond using surfer slang.

                                    Chat History: {chat_history}
                                    Context: {context}
                                    Question: {question}
                                 """,
                        input_variables=["chat_history", "context", "question"])

# create a new instance of the LLMChain class
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True, verbose=True)

# assign the LLMChain instance to the chat_chain variable
chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory)

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
while True:
    question = input("> ")
    response = chat_chain.invoke({"context": current_weather, "question": question})

    print(response["text"])