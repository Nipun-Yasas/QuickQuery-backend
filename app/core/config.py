import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # AWS
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    
    # AI & Vector DB
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI")

settings = Settings()