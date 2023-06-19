from langchain import PromptTemplate

template = """
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate.from_template(template)
response = prompt.format(product="colorful socks")

print(response)

# ----------------------------------------------------------------

# An example prompt with no input variables
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
print(no_input_prompt.format())
# -> "Tell me a joke."

# An example prompt with one input variable
one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
print(one_input_prompt.format(adjective="funny"))
# -> "Tell me a funny joke."

# An example prompt with multiple input variables
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}."
)
print(multiple_input_prompt.format(adjective="funny", content="chickens"))
# -> "Tell me a funny joke about chickens."

# ----------------------------------------------------

template = "Tell me a {adjective} joke about {content}."

prompt_template = PromptTemplate.from_template(template)
print(prompt_template.input_variables)
# -> ['adjective', 'content']

prompt_template.format(adjective="funny", content="chickens")
# -> Tell me a funny joke about chickens.
