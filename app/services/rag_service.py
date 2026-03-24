from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from app.core.config import settings
import tempfile
import os

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GOOGLE_API_KEY,
    output_dimensionality=768, 
    task_type="retrieval_document" 
)

def index_document_to_pinecone(file_content, file_name, user_id):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update({"user_id": user_id, "source": file_name})

        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name="quickquery-rag-docs",
            namespace=user_id
        )
        return True
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0
)

def query_document(question, user_id):
    vectorstore = PineconeVectorStore(
        index_name="quickquery-rag-docs",
        embedding=embeddings,
        namespace=user_id
    )

    docs = vectorstore.similarity_search(question, k=5)
    
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = """
    You are a helpful assistant for QuickQuery. Use the following context to answer the user's question accurately.
    If the context doesn't contain the answer, just say you don't know based on the documents.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = prompt_template.format(context=context, question=question)

    response = llm.invoke(prompt)
    return response.content