from fastapi import FastAPI , UploadFile , File , HTTPException
import logging
from botocore.exceptions import NoCredentialsError
from app import data_access as S3_FILE
from app.scripts import pdf_scrape as PDF_SCRAPE
# from app.scripts import web_scrape as WEB_SCRAPE
from app import constant as CONSTANT
from app.models import  ChatRequest , ChatResponse
from app.views import generate_rag_response

app = FastAPI()

logger = logging.getLogger(__name__)

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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receive user query and return the RAG answer with S3 source URLs.
    """
    try:
        logger.info("Received chat request")

        result = generate_rag_response(query=request.query)
        sources = []
        for source in result["sources"]:
            source = source.copy()
            s3_key = source.get("s3_key")

            if s3_key:
                source["url"] = S3_FILE.s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": S3_FILE.BUCKET_NAME,
                        "Key": s3_key,
                        "ResponseContentDisposition": "inline",
                    },
                    ExpiresIn=3600,
                )
            sources.append(source)
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            status="successfully retrieved",
        )

    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to generate chat response")