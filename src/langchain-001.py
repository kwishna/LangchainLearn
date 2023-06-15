import dotenv, openai
from langchain import LLMChain
from langchain.llms import OpenAI

dotenv.load_dotenv()

# The most basic building block of LangChain is calling an LLM on some input.
_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)
print(_open_ai_llm('What could be simple yet impactful name for my new ladies fashion shop?'))
# Fashionista Muse.

