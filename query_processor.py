import os
import dotenv
dotenv.load_dotenv()

from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FinRAG"

client = MongoClient(os.getenv('MONGO_URI'))
db_name = 'FinRAG'
collection_name = 'Vector-Store-FinRAG'
collection = client[db_name][collection_name]

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                          google_api_key=os.getenv('GOOGLE_API_KEY'),
                                          task_type='retrieval_query')
atlas_search_index = "FinRAG-search-index"
vectorstore = MongoDBAtlasVectorSearch(collection=collection,
                                       embedding=embeddings,
                                       index_name=atlas_search_index,
                                       relevance_score_fn="cosine")

class Generator:
    def __init__(self) -> None:
        self.__llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.__retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        self.__out_parser = StrOutputParser()
        # self.__chat_history = []
        # self.__summary = None

    def __retrieve_documents(self, query : str):
        return self.__retriever.invoke(query)

    def __generator(self, query : str):
        docs = self.__retrieve_documents(query)

        prompt = ChatPromptTemplate([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"),
            ("user", "Question: {question}\nContext: {context}")
        ])

        rag_chain = prompt | self.__llm | self.__out_parser

        context = "\n".join([doc.page_content for doc in docs])
        print(context)
        print("----")
        print(docs)
        answer = rag_chain.invoke({"question": query, "context": context})
        return answer

    def process(self, query: str):  # Call will be made to this function to generate answers
        answer = self.__generator(query)
        return answer

