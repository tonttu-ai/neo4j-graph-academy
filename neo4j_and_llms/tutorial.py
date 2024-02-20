import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import Prompt
from langchain.prompts import PromptTemplate

# load the .env file
load_dotenv()

# load a prompt template
with open("./prompt_template_cockney_fruit_seller.txt", "r") as file:
    prompt_text = file.read()

# create a new prompt
template = PromptTemplate(template=prompt_text, input_variables=["fruit"])
template.save("cockney_fruit_seller.json")

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = llm.invoke(template.format(fruit="peaches"), model="gpt-3.5-turbo-instruct", max_tokens=100, temperature=0.9, top_p=1, frequency_penalty=0, presence_penalty=0)

print(response)