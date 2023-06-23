import os
import textwrap
import openai
import dotenv
from langchain import OpenAI
from llama_index import LLMPredictor, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex

dotenv.load_dotenv()

openai.api_key = "ENTER_YOUR_OPENAI_KEY_HERE"
data_directory = os.getcwd() + "/data"


def extract_data():
    llm_predictor = LLMPredictor(llm=OpenAI(model_name="text-davinci-003"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    dirs = os.listdir(data_directory)
    assert len(dirs) > 0

    for filename in dirs:
        if os.path.isfile(os.path.join(data_directory, filename)):
            documents = SimpleDirectoryReader(input_files=[f"{data_directory}/{filename}"]).load_data()
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

            prompt = "Describe History and Evolution of Cybercrime."

            query_engine = index.as_query_engine()
            response = query_engine.query(prompt)
            print(response)
            print(textwrap.fill(str(response), 100))


if __name__ == '__main__':
    extract_data()
