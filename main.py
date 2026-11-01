from fastapi import FastAPI , UploadFile , File , HTTPException , Request , status
from pydantic import HttpUrl
import logging
import os
import chromadb
import asyncio
from botocore.exceptions import NoCredentialsError
from app import data_access as S3_FILE
from app.scripts import pdf_scrape as PDF_SCRAPE
# from app.scripts import web_scrape as WEB_SCRAPE
from app import constant as CONSTANT
from app.models import  ChatRequest , ChatResponse
from app.views import generate_rag_response
from app.scripts import web_scrape as WEB_SCRAPE
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
  try:
    filename = file.filename.lower().strip('")\' ')
    if filename.endswith(CONSTANT.VALID_FILES):
      s3_file_key = f"{S3_FILE.S3_DIRECTORY}{file.filename}"
      client = S3_FILE.s3_client
      client.upload_fileobj(file.file, S3_FILE.BUCKET_NAME, s3_file_key)
      logger.info("File uploaded successfully to S3: %s", S3_FILE.BUCKET_NAME)
      return {
          "status": "success",
          "message": "File uploaded successfully to S3",
          "filename": file.filename,
      }
    logger.error("Failed to upload file: Invalid file type -> %s", file.filename)
    return {
        "status": "failed",
        "message": "Failed to Upload: Invalid file type",
        "filename": file.filename,
    }
  except NoCredentialsError:
    logger.exception("AWS credentials not found or invalid.")
    raise HTTPException(status_code=500, detail="AWS credentials not found or invalid.")
  except Exception as e:
    logger.exception("file upload failed")
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-file")
async def extract_file():
  try:
    bucket_name = S3_FILE.BUCKET_NAME  
    s3_directory = S3_FILE.S3_DIRECTORY.strip("/")
    s3_folder_path = f"s3://{bucket_name}/{s3_directory}/"
    logger.info("Getting s3 path for the files")
    extractor = PDF_SCRAPE.PDFExtractor(s3_folder_path)
    logger.info("Extracting files in s3")
    extractor.store_to_chroma()
    logger.info("PDFExtractor completed successfully.")
    return {
        "status": "success",
        "message": "Data extracted and stored to Chroma successfully.",
    }
  except Exception:
    logger.error("Extraction failed")
    raise HTTPException(status_code=500, detail="There is no file path")

@app.post("/upload-website")
async def upload_website_scrape(url:HttpUrl):
    crawler = WEB_SCRAPE.WebScrapping(source_url=str(url))
    scraped_data = crawler.scrape_webpages()
    crawler.chunk_and_store_embeddings(scraped_data)
    logger.info("The Uploaded URL is extracted and stored to the DB.")
    return {"status": "success", "url": str(url)}
   

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Receive user query and return the RAG answer as an SSE stream.
    Monitors client disconnection to cleanly stop token generation.
    """
    logger.info(f"Received chat request: {request.query}")

    try:
        async def event_publisher():
            try:
                async for chunk in generate_rag_response(query=request.query):

                    if await req.is_disconnected():
                        logger.info(f"Client disconnected; aborting generation for query: {request.query}")
                        break
                    yield chunk
            except asyncio.CancelledError:
                logger.info("Client stream task cancelled.")
                raise

        return StreamingResponse(
            event_publisher(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disables proxy buffering in Nginx
            },
        )
    except Exception as e:
        logger.exception(f"Failed to initialize chat stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate chat stream")

@app.delete("/delete-collection")
def delete_chroma_collection():
    try:
        client = chromadb.CloudClient(api_key=os.getenv("CHROMA_API_KEY"),tenant=os.getenv("TENANT"),database=os.getenv("DATABASE"))
        collection_name = CONSTANT.CHROMA_COLLECTIONS
        collection = client.get_collection(name=collection_name)
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
        return {"status": "success", "message": f"Collection '{collection_name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

