### Installations 
# pip install faiss-cpu dotted-dict python-dotenv streamlit
### Importing Libraries 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
from dotted_dict import DottedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from datetime import datetime
### Configuration
load_dotenv(find_dotenv())
azure_config = {
    "model_deployment": os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT_NAME'),
    "embedding_deployment": os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
    "embedding_name": os.getenv('AZURE_OPENAI_EMBEDDING_NAME'),
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "api_version": os.getenv('AZURE_OPENAI_API_VERSION'),
    "endpoint": os.getenv('AZURE_OPENAI_BASE_URL'),
    "model": os.getenv('AZURE_OPENAI_MODEL_NAME')
}
### Model Creation
print("creating models")
models=DottedDict()
llm = AzureChatOpenAI(temperature=0,
                    api_key=azure_config["api_key"],
                    openai_api_version=azure_config["api_version"],
                    azure_endpoint=azure_config["endpoint"],
                    model=azure_config["model_deployment"],
                    validate_base_url=False)
embedding_model = AzureOpenAIEmbeddings(
    api_key=azure_config["api_key"],
    openai_api_version=azure_config["api_version"],
    azure_endpoint=azure_config["endpoint"],
    model = azure_config["embedding_deployment"]
)
models.llm=llm
models.embedding_model=embedding_model 
### Linking the SSL Certificate 
os.environ['REQUESTS_CA_BUNDLE'] = 'C:\One-Drive\OneDrive - Tredence\Documents\LLM Engineer Track Program\Capstone_Project\Zscaler Root CA.crt'
### Extracting Data from the URL 
loader = WebBaseLoader(web_path='https://en.wikipedia.org/wiki/FIFA_World_Cup')
data = loader.load()
#load_string=str(data)
### Filtering out last 5 years' Data 
current_year = datetime.now().year
# List of recent World Cup years starting from 2022
world_cup_years = [2022, 2018, 2014, 2010, 2006]

# Adjust the list if the current year is past 2022
if current_year > 2022:
    latest_wc_year = 2022 + ((current_year - 2022) // 4) * 4
    world_cup_years = [latest_wc_year - i*4 for i in range(5)]

filtered_data = []
for doc in data:
    if any(f'{year} FIFA World Cup' in doc.page_content for year in world_cup_years):
        filtered_data.append(doc)
world_cup_years
### Chunking 
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter =RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, is_separator_regex=False, separators=["."])
chunks = text_splitter.split_documents(filtered_data)
chunks
### Converting to Vector Embeddings 
embeddings = models.embedding_model
vectordb=FAISS.from_documents(chunks,embeddings)
### Prompt Engineering 
prompt_template = """
You are an AI language model. Answer the questions based solely on the context provided below. 
Do not use any external information or prior knowledge. 
If questions ask for data beyond the last 5 FIFA world cups, refuse to answer.
Context:\n {context}?\n
Question: \n{question}\n
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(models.llm, chain_type="stuff", prompt=prompt)
### Querying
# st.title("FIFA WORLD CUP RAG")
# @st.cache_resource(show_spinner=False)
def get_answer(query):
    document_search=vectordb
    similar_docs = document_search.similarity_search(query, k=1) # get closest chunks
    answer = chain.invoke(input={"input_documents": similar_docs, "question": query})
    return answer
answer = get_answer('Name some FIFA World Cup winners in the last 10 years.')
answer
### Streamlit
# with st.form("my_form"):
#     query = st.text_area("Ask a Question about the last 5 FIFA World Cups.")
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         response = get_answer(query)
#         st.write(response['output_text'])

user_input = st.text_area("Enter your prompt:")

if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            response = get_answer(user_input)
            st.write(response)
    else:
        st.warning("Please enter a prompt.")