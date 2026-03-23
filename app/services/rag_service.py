from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
import tempfile
import os

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") # Automatically 768 dimensions

def index_document_to_pinecone(file_content, file_name, user_id):
    # 2. Create a temporary file to read the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        # 3. Load and Split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # 4. Add metadata
        for chunk in chunks:
            chunk.metadata.update({"user_id": user_id, "source": file_name})

        # 5. Upload to Pinecone (Requires a 768-dimension index!)
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name="quickquery-rag-docs",
            namespace=user_id
        )
        return True
    
    finally:
        # 6. Clean up the temp file from your hard drive
        if os.path.exists(tmp_path):
            os.remove(tmp_path)