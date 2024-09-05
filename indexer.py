import os
from dotenv import load_dotenv
load_dotenv()

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


# only these filetypes will be allowed to parsed
allowed_filetypes = ['pdf', 'doc', 'docx', 'txt', 'xlsx', 'csv', 'ppt', 'pptx', 'md']

# A smart parser for files
parser = LlamaParse(
    api_key=os.getenv('LLAMA_INDEX'),
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

def check_filetype(filename : str):
    """Checks if the filetype is allowed to be parsed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_filetypes


def parsed_segmentation(file_path : str,
                        chunk_size : int,
                        chunk_overlap : int) -> list[Document]:
    """
    Parse a file's content and segment it into chunks.

    :param file_path: Path to the file to be parsed.
    :param chunk_size: Size of each chunk in characters.
    :param chunk_overlap: Amount of overlap between consecutive chunks.
    :return: A list of Document objects representing segmented content.
    """
    file = parser.load_data(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text = "".join([doc.text for doc in file])
    chunks = text_splitter.split_text(text)
    source = os.path.basename(file_path)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                'source': source,
                'chunk_index': source + str(idx)
            }
        ) for idx, chunk in enumerate(chunks)
    ]
    return documents


def find_file(source : str,
              vectorStore : MongoDBAtlasVectorSearch) -> bool:
    """
    Check if the document with the given filename already exists in the vector store.

    :param filename: The file name of the file to check.
    :param vectorStore: The MongoDBAtlasVectorSearch instance to query
    :return: True if the file exits, False otherwise
    """
    query_result = vectorStore.similarity_search(f"metadata.source:{source}")
    return len(query_result) > 0


client = MongoClient(os.getenv('MONGO_URI'))
try:
    client.admin.command('ping')
    print('Pinged you deployment. You successfully connected to MongoDB!')
except Exception as e:
    print(e)
db_name = 'FinRAG'
collection_name = 'Vector-Store-FinRAG'
collection = client[db_name][collection_name]

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                          google_api_key=os.getenv('GOOGLE_API_KEY'),
                                          task_type='retrieval_document')

data_directory = 'data'
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
vectorstore = MongoDBAtlasVectorSearch(collection, embeddings)

for filename in os.listdir(data_directory):
    try:
        if not check_filetype(filename):
            raise RuntimeError(f"{filename} is not allowed to be parsed")
    except RuntimeError as e:
        print(e)
        print(f"Allowed filetypes are: {allowed_filetypes}")

    if find_file(source=filename, vectorStore=vectorstore):
        print(f"Document with file name `{filename}` already exists in the vectorstore. Skipping upsertion")
        continue

    file_path = os.path.join(data_directory, filename)

    # parse and chunk the data
    documents = parsed_segmentation(file_path=file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # create embeddings and upsert to the vectorstore
    vectorStore = MongoDBAtlasVectorSearch.from_documents(
        documents,
        embeddings,
        collection=collection
    )

