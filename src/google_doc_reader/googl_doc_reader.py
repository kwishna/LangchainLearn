from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from llama_hub.google_docs.base import GoogleDocsReader

# load documents
gdoc_ids = ['1wf-y2pd9C878Oh-FmLH7Q_BQkljdm6TQal-c1pUfrec']
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)
langchain_documents = [d.to_langchain_format() for d in documents]

# initialize sample QA chain
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm)
question = "Write the summary."
answer = qa_chain.run(input_documents=langchain_documents, question=question)
