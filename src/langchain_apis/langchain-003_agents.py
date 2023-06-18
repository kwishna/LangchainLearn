import dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

dotenv.load_dotenv()

# https://python.langchain.com/docs/get_started/quickstart#agents

# --------- Using LLM ----------------

# Loading the language model we're going to use to control the agent.
_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)

# Load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=_open_ai_llm)

# Initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, _open_ai_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
print(agent.run("What is the air pollution index level in Bangalore, India?"))

# Entering new  chain...
# I need to find current data on air pollution levels in Bangalore
# Action: Search
# Action Input: Air pollution index level in Bangalore, India
# Observation: It comes in ranked at 82nd of the most polluted cities in India, with a 2019 PM2.5 rating of 32.6 µg/m³.
# PM2.5 stands for particulate matter ...
# Thought: I now know the air pollution index level in Bangalore, India
# Final Answer: The air pollution index level in Bangalore, India is 32.6 µg/m³.
#
# > Finished chain.

# The air pollution index level in Bangalore, India is 32.6 µg/m³.

# ----------------- Using Chat Models --------------------

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
chat = ChatOpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
print(response)
