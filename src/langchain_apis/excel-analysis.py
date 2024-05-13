import pandas
import pandasai
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_openai.llms import OpenAI

load_dotenv()

llm = OpenAI()
llm.model_name = "gpt-3.5-turbo-instruct"

print("--------------------------------------------------------")

# df = pandas.read_csv("./sat_results.csv")
# sdf = pandasai.SmartDataframe(df, config={"llm": llm})
#
# sdf.verbose = True
# ans = sdf.chat("How many unique 'DBN' are there?")
# print(ans)

print("--------------------------------------------------------")

agent = create_csv_agent(llm=llm, path=['./sat_results.csv'], verbose=True)
res = agent.invoke({"input": "How many rows of data do you have?"})
print(res)