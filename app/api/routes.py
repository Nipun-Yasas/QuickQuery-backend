from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.s3_service import upload_file_to_s3
from app.services.rag_service import index_document_to_pinecone, query_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Mock User ID (Replace with Clerk User ID later)
    user_id = "test_user_123"
    
    file_content = await file.read()
    
    # 1. Upload to S3
    s3_url = upload_file_to_s3(file_content, file.filename)
    
    # 2. Index in Pinecone (RAG)
    index_document_to_pinecone(file_content, file.filename, user_id)
    
    return {
        "status": "success",
        "s3_url": s3_url,
        "message": "Document indexed and stored."
    }

# 3. Model for asking a question
class QueryRequest(BaseModel):
    user_id: str
    question: str

@router.post("/query")
async def ask_question(request: QueryRequest):
    # Retrieve the answer using our RAG service
    answer = query_document(request.question, request.user_id)
    
    return {
        "status": "success",
        "answer": answer
    }