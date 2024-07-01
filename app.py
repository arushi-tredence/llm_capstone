from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
from dotted_dict import DottedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from datetime import datetime
os.environ['CURL_CA_BUNDLE'] = ''
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
loader = WebBaseLoader(web_path='https://en.wikipedia.org/wiki/FIFA_World_Cup')
data = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter =RecursiveCharacterTextSplitter(chunk_size=25, 
                                              chunk_overlap=10, 
                                              length_function=len, 
                                              is_separator_regex=False, 
                                              separators=["."])
chunks = text_splitter.split_documents(data)
### Filtering Chunks
text_data = " ".join([doc.page_content for doc in data])
current_year = datetime.now().year
query = f"Last 5 FIFA World Cup Tournaments from {current_year}"
world_cup_years = [2022, 2018, 2014, 2010, 2006]
if current_year > 2022:
    latest_wc_year = 2022 + ((current_year - 2022) // 4) * 4
    world_cup_years = [latest_wc_year - i*4 for i in range(5)]
def contains_world_cup_year(chunk, years):
    return any(str(year) in chunk.page_content for year in years)

filtered_chunks = [chunk for chunk in chunks if contains_world_cup_year(chunk, world_cup_years)]
for chunk in filtered_chunks:
    print(chunk)
embeddings = models.embedding_model
vectordb=FAISS.from_documents(filtered_chunks,embeddings)
prompt_template = """
It is currently the year: {current_year}
Answer the questions about the given World Cup Years only. 
Make sure you answer only from the given context.
World Cup Years: \n {world_cup_years} \n
Context:\n {context}?\n
Question: \n{question}\n
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["current_year", "world_cup_years", "context", "question"])
chain = load_qa_chain(models.llm, chain_type="stuff", prompt=prompt)
st.title("FIFA World Cup RAGBot")
@st.cache_resource(show_spinner=False)
def get_answer(query):
    document_search=vectordb
    similar_docs = document_search.similarity_search(query, k=1) # get closest chunks
    answer = chain.invoke(input={"current_year": current_year, "world_cup_years": world_cup_years, "input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer

user_input = st.text_area("Enter your prompt:")
if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            response = get_answer(user_input)
            st.write(response)
    else:
        st.warning("Please enter a prompt.")