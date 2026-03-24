from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
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

# 7. Initialize the LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0
)

def query_document(question, user_id):
    # 1. Access the specific user's namespace in Pinecone
    vectorstore = PineconeVectorStore(
        index_name="quickquery-rag-docs",
        embedding=embeddings,
        namespace=user_id
    )

    # 2. Search for relevant context (k=5 means top 5 chunks)
    docs = vectorstore.similarity_search(question, k=5)
    
    # 3. Join the text from the retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # 4. Create a prompt for Gemini
    prompt_template = """
    You are a helpful assistant for QuickQuery. Use the following context to answer the user's question accurately.
    If the context doesn't contain the answer, just say you don't know based on the documents.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = prompt_template.format(context=context, question=question)

    # 5. Generate and return the answer
    response = llm.invoke(prompt)
    return response.content