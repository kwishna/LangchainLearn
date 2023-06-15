import dotenv
from langchain import OpenAI, ConversationChain

dotenv.load_dotenv()

# Loading the language model we're going to use to control the agent.
_open_ai_llm = OpenAI(temperature=0.9, model_name='text-davinci-003', max_tokens=256, top_p=1)

# the ConversationChain has a simple type of memory that remembers all previous inputs/outputs
# and adds them to the context that is passed.
conversation = ConversationChain(llm=_open_ai_llm, verbose=True)

output = conversation.predict(input="Hi, I know you. You're Mr. Bean! The comedian.")
print(output)
# Hi there! I'm not Mr. Bean, the comedian, but I am an AI.
# I'm designed to answer questions about a variety of topics. What can I help you with today?

# Output:-
# > Entering new  chain...
# Prompt after formatting:
# The following is a friendly conversation between a human and an AI.
# The AI is talkative and provides lots of specific details from its context.
# If the AI does not know the answer to a question, it truthfully says it does not know.
#
# Current conversation:
#
# Human: Hi, I know you. You're Mr. Bean! The comedian.
# AI:
#
# > Finished chain.
#  Hi there! I'm not Mr. Bean, the comedian, but I am an AI.
#  I'm designed to answer questions about a variety of topics. What can I help you with today?


output = conversation.predict(input="Anyway, Repeat your previous message a little more funnier.")
print(output)
#  Hi there! I'm not Mr. Bean, the comedian,
#  but I could make you smile if you have a question I can answer for you. What can I do for you today

# Output:-
# > Entering new  chain...
# Prompt after formatting:
# The following is a friendly conversation between a human and an AI.
# The AI is talkative and provides lots of specific details from its context.
# If the AI does not know the answer to a question, it truthfully says it does not know.
#
# Current conversation:
# Human: Hi, I know you. You're Mr. Bean! The comedian.
# AI:  Hi there! I'm not Mr. Bean, the comedian, but I am an AI.
# I'm designed to answer questions about a variety of topics. What can I help you with today?
#
# Human: Anyway, Repeat your previous message a little more funnier.
# AI:
#
# > Finished chain.
#  Hi there! I'm not Mr. Bean, the comedian,
#  but I could make you smile if you have a question I can answer for you. What can I do for you today?

