import os
import yaml
import logging
from groq import Groq  
from .retriever import HybridRetrieval 
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

retriever = HybridRetrieval()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../prompts.yaml")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

prompt_template = prompts["rag_chat_prompt"]

def generate_rag_response(query: str, top_k: int = 3) -> dict:
    """
    Executes hybrid search, formats context via prompt template, 
    and sends the payload to Groq's chat completion API.
    """
    try:
        logger.info(f"Running hybrid search for query: {query}")
        search_results = retriever.hybrid_search(query=query, top_k=top_k)

        if not search_results:
            return {
                "answer": "I cannot find the answer in the provided documents.",
                "sources": []
            }

        context_blocks = []
        sources = []
        for hit in search_results:
            context_blocks.append(f"[Source ID: {hit['id']}]\n{hit['text']}")
            sources.append({
                "id": hit["id"],
                "source_type": hit["source"],
                "score": hit["score"]
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)
        logger.info(formatted_context)

        final_prompt = prompt_template.format(
            context=formatted_context,
            query=query
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            model="openai/gpt-oss-20b",
            temperature=0.1
        )

        answer_text = chat_completion.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in generate_rag_response: {str(e)}")
        raise e