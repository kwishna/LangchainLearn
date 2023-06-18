import dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

# In the previous example, the text we passed to the model contained instructions to generate a company name.
# For our application, it'd be great if the user only had to provide the description of a company/product,
# without having to worry about giving the model instructions.

# -------------- Using LLM ----------------

_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)
_prompt = PromptTemplate(
    input_variables=["actor"],
    template="Imagine a deadly name for {actor} in negative role?",
)

print(_prompt.format(actor="batman"))

_chain = LLMChain(llm=_open_ai_llm, prompt=_prompt)
print(_chain.run("batman"))  # Nightmare Batman.

# ----------------------------------- Using Chat Models -----------------------------

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.7)

# System template Or SystemMessagePromptTemplate is basically a pre-condition.
system_template = "You're a movie maniac. Suggest a new name of movie if it were in {genre} genre and {language} language."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human template or HumanMessagePromptTemplate is the human prompt that will be sent to AI model.
human_template = "{movie_name}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#  ChatPromptTemplate is the final prompt to be sent to AI model.
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

messages = chat_prompt_template.format_messages(
    genre="comedy",
    language="hindi",
    movie_name="The dark Knight."
)
# System message is the objective that the AI model should follow.
print(messages)
# [SystemMessage(content="You're a movie maniac. Suggest a new name of movie if it were in comedy genre and hindi language.",
# additional_kwargs={}), HumanMessage(content='The dark Knight.', additional_kwargs={}, example=False)]

chain = LLMChain(llm=chat, prompt=chat_prompt_template)
response = chain.run(genre="comedy", language="hindi", movie_name="The dark Knight.")
print(response)

# AIMessage is a message sent from the perspective of the AI the human is interacting with.
# HumanMessage is a message sent from the perspective of the human.
# SystemMessage is a message setting the objectives the AI should follow.



