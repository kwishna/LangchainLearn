import os
import openai
from dotenv import load_dotenv

load_dotenv()  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# -------- 1 --------
def get_completion(prompt, model="gpt-3.5-turbo"):
    _messages = [{"role": "user", "content": prompt}]
    _response = openai.ChatCompletion.create(
        model=model,
        messages=_messages,
        temperature=0,
    )
    return _response.choices[0].message["content"]


print(get_completion("Q. What is 1+1?\n A."))
print("-"*20)
print(get_completion("What is 1+1?"))

# ------------- 2 ----------

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

_prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print("The prompt is: ", _prompt)

response = get_completion(_prompt)

print(response)
