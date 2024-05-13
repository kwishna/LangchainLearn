import os, pandasai, pandas
from langchain_community.llms import OpenAI
from langchain_openai.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()
llm.model_name="gpt-3.5-turbo-instruct"

df = pandas.read_csv("./sat_results.csv")
sdf = pandasai.SmartDataframe(df, config={"llm": llm})

sdf.verbose = True
ans = sdf.chat("How many unique 'DBN' are there?")
print(ans)