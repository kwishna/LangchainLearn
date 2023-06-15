import dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)
_prompt = PromptTemplate(
    input_variables=["superhero"],
    template="Imagine a deadly name for {superhero} in negative role?",
)
# 1
print(_prompt.format(superhero="batman"))

# 2
_chain = LLMChain(llm=_open_ai_llm, prompt=_prompt)
print(_chain.run("batman")) # Nightmare Batman.
