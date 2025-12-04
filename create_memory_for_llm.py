import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data_path):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(data_path, file)
            loader = PyPDFLoader(file_path)
            # Load documents and split pages with metadata
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file  # Add filename to metadata
                # The page number is included under 'page' in metadata
        documents.extend(docs)
    return documents

documents = load_pdf_files(DATA_PATH)
print("Number of documents loaded:", len(documents))

# Print metadata for the first few documents to verify the keys
for doc in documents[:5]:
    print(doc.metadata)

# Step 2: Create Chunks and Preserve Metadata
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # Split documents while preserving metadata
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
print("Number of text chunks created:", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"Vector store saved to {DB_FAISS_PATH}")
