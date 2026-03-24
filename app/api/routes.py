from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.s3_service import upload_file_to_s3
from app.services.rag_service import index_document_to_pinecone, query_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    user_id = "test_user_123"
    
    file_content = await file.read()
    
    s3_url = upload_file_to_s3(file_content, file.filename)
    
    index_document_to_pinecone(file_content, file.filename, user_id)
    
    return {
        "status": "success",
        "s3_url": s3_url,
        "message": "Document indexed and stored."
    }

class QueryRequest(BaseModel):
    user_id: str
    question: str

@router.post("/query")
async def ask_question(request: QueryRequest):
    answer = query_document(request.question, request.user_id)
    
    return {
        "status": "success",
        "answer": answer
    }