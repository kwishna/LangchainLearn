import dotenv
from langchain.llms import OpenAI

dotenv.load_dotenv()

# We're building an application that generates a company name based on a company description using OpenAI LLM model.
# The most basic building block of LangChain is calling an LLM on some input.

# ----------------- Using LLM -----------------
_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)
print(_open_ai_llm('What could be simple yet impactful name for my new ladies fashion shop?'))
# Fashionista Muse.

# ------------------------------------- --------------------
# Building the same application using langchain Chat Models.
# While chat models use language models under the hood,
# the interface they expose is a bit different:
# rather than expose a "text in, text out" API,
# they expose an interface where "chat messages" are the inputs and outputs.

# ------------ Using Chat Models ------------------

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)

chat = ChatOpenAI(temperature=0.7)
chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# >> AIMessage(content="J'aime programmer.", additional_kwargs={})

chat.predict("Translate this sentence from English to French. I love programming.")
# >> J'aime programmer
