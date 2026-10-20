import time
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import chromadb
import logging
import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import constant as CONSTANT
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

# if not cloud chroma used 

# admin_client = chromadb.AdminClient()
# try:
#     admin_client.get_tenant(name=CONSTANT.CHROMA_TENANT)
# except Exception:
#     admin_client.create_tenant(name=CONSTANT.CHROMA_TENANT)

# try:
#     admin_client.get_database(
#         name=CONSTANT.CHROMA_DB,
#         tenant=CONSTANT.CHROMA_TENANT
#     )
# except Exception:
#     admin_client.create_database(
#         name=CONSTANT.CHROMA_DB,
#         tenant=CONSTANT.CHROMA_TENANT
#     )


class WebScrapping:
    def __init__(self, source_url: str, delay: float = 2.0):
        self.source_url = source_url
        self.delay = delay

        parsed_source_url = urlparse(source_url)
        self.base_domain = parsed_source_url.netloc

        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")

    def _crawl_site_links(self):
        visited_urls = set()
        queue = [self.source_url]
        discovered_set = set()

        driver = webdriver.Chrome(options=self.chrome_options)

        try:
            while queue:
                current_url = queue.pop(0)
                if current_url in visited_urls:
                    continue
                visited_urls.add(current_url)
                discovered_set.add(current_url)
                try:
                    driver.get(current_url)
                    time.sleep(self.delay)
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                    for tag in soup.find_all('a', href=True):
                        raw_href = tag['href']
                        absolute_url = urljoin(current_url, raw_href)
                        parsed_link = urlparse(absolute_url)

                        if parsed_link.netloc == self.base_domain:
                            clean_url = absolute_url.split('#')[0]
                            if not clean_url.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.zip', '.css', '.js', '.ico', '.svg')):
                                if clean_url not in visited_urls and clean_url not in queue:
                                    queue.append(clean_url)

                except Exception as e:
                    logger.info(f"crawling link {current_url}: {e}")

        finally:
            driver.quit()
        return discovered_set

    def scrape_webpages(self):
        web_pages = self._crawl_site_links()

        scraped_data = []
        driver = webdriver.Chrome(options=self.chrome_options)

        try:
            for url in sorted(web_pages):
                try:
                    logger.info(f"Scraping page: {url}")
                    driver.get(url)
                    time.sleep(self.delay)
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    for tag in soup(CONSTANT.DECOMPOSE_TAGS):
                        tag.decompose()

                    title_elements = soup.find(CONSTANT.TITLE_TAGS)
                    title = title_elements.get_text(strip=True) if title_elements else "Untitled Page"

                    content_elements = soup.find_all(CONSTANT.MAIN_CONTENT)
                    page_text_chunks = []
                    for element in content_elements:
                        text = element.get_text(separator=" ", strip=True)
                        if text and len(text) > 20:
                            page_text_chunks.append(text)
                            
                    page_text = "\n".join(page_text_chunks)
                    if page_text:
                        scraped_data.append({
                            "title": title,
                            "page_text": page_text,
                            "source": url
                        })
                        logger.info(f"Extracted {len(page_text)} characters from {url}")
                    else:
                        logger.warning(f"No text found on page: {url}")
                        
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
        finally:
            driver.quit()
        return scraped_data

    def chunk_and_store_embeddings(self, scraped_data: list, persist_directory=CONSTANT.CHROMA_DIRS, tenant=CONSTANT.CHROMA_TENANT, database=CONSTANT.CHROMA_DB):
        logger.info("Initializing Hugging Face embeddings and text chunker...")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=600,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        all_metadatas = []
        all_ids = []  

        for doc in scraped_data:
            title = doc.get("title", "Untitled Page")
            page_text = doc.get("page_text", "")
            source = doc.get("source", "")

            if not page_text.strip():
                continue

            chunks = text_splitter.split_text(page_text)

            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "title": title,
                    "source_url": source,
                    "chunk_index": i
                })

        if all_chunks:
            logger.info(f"Generating embeddings and storing {len(all_chunks)} chunks into ChromaDB...")
            
            chunk_embeddings = embeddings.embed_documents(all_chunks)
            
            # client = chromadb.PersistentClient(
            #     path=persist_directory,
            #     settings=chromadb.Settings(tenant=tenant, database=database)
            # )
            client = chromadb.CloudClient(api_key=os.getenv("CHROMA_API_KEY"),tenant=os.getenv("TENANT"),database=os.getenv("DATABASE"))
            collection_name = CONSTANT.CHROMA_COLLECTIONS
            collection = client.get_or_create_collection(name=collection_name)

            collection.add(
                ids=all_ids,
                documents=all_chunks,
                embeddings=chunk_embeddings,
                metadatas=all_metadatas
            )
            logger.info("Successfully embedded and stored all chunks into ChromaDB!")
        else:
            logger.warning("No text chunks generated for ingestion.")


if __name__ == "__main__":
    url = input("Enter website URL: ").strip()
    crawler = WebScrapping(source_url=url)
    scraped_data = crawler.scrape_webpages()
    crawler.chunk_and_store_embeddings(scraped_data)
