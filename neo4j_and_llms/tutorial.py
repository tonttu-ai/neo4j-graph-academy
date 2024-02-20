import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

# load the .env file
load_dotenv()

# load a prompt template
with open("./prompt_template_cockney_fruit_seller.txt", "r") as file:
    prompt_text = file.read()

# create a new prompt
template = PromptTemplate(template=prompt_text, input_variables=["fruit"])
template.save("cockney_fruit_seller.json")

# create a new LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
             model="gpt-3.5-turbo-instruct", 
             max_tokens=100, 
             temperature=0.9, # cranks up the creativity, more suitable to a cockney fruit seller
             top_p=1,
             frequency_penalty=0, 
             presence_penalty=0)

# create a new LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser=SimpleJsonOutputParser()
)

response = llm_chain.invoke({"fruit": "apples"})

print(response)