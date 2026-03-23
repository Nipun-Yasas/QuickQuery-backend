from fastapi import APIRouter, UploadFile, File
from app.services.s3_service import upload_file_to_s3
from app.services.rag_service import index_document_to_pinecone

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