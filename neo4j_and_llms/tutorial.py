import dotenv
from langchain_openai import OpenAI
from langchain.prompts import Prompt

# load a prompt template
with open("./prompt_template_cockney_fruit_seller.txt", "r") as file:
    prompt_template = file.read()

template = Prompt(template = prompt_template, input_variables=["fruit"])

llm = OpenAI()

response = llm.invoke(template.format(fruit="peaches"), model="gpt-3.5-turbo-instruct", max_tokens=100, temperature=0.9, top_p=1, frequency_penalty=0, presence_penalty=0)

print(response)