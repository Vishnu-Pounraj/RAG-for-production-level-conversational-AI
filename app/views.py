import os
import yaml
import logging
from groq import Groq  
from .retriever import HybridRetrieval 
from dotenv import load_dotenv
import json
from typing import AsyncGenerator
from groq import AsyncGroq
import asyncio
import re

load_dotenv()

logger = logging.getLogger(__name__)

retriever = HybridRetrieval()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../prompts.yaml")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

prompt_template = prompts["rag_chat_prompt"]

groq_client = AsyncGroq()

async def generate_rag_response(query: str) -> AsyncGenerator[str, None]:
    """
    Executes hybrid search, formats context via prompt template, 
    and yields SSE formatted chunks (streaming tokens, then used sources).
    """
    try:
        logger.info(f"Running hybrid search for query: {query}")

        if asyncio.iscoroutinefunction(getattr(retriever, "hybrid_search", None)):
            search_results = await retriever.hybrid_search(query=query)
        else:
            search_results = await asyncio.to_thread(retriever.hybrid_search, query=query)

        if not search_results:
            fallback_msg = "I cannot find the answer in the provided documents."
            yield f"data: {json.dumps({'type': 'token', 'text': fallback_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context = []
        source_map = {}

        for idx, hit in enumerate(search_results, start=1):
            source_data = hit.get("source") or {}

            if isinstance(source_data, dict):
                source_url = source_data.get("source_url", "").strip()
                document_name = (
                    source_data.get("document_name")
                    or source_data.get("source")
                    or f"Document-{idx}"
                )
                section = source_data.get("section", "")
            elif isinstance(source_data, str):
                source_url = source_data.strip()
                document_name = f"Document-{idx}"
                section = ""
            else:
                source_url = ""
                document_name = f"Document-{idx}"
                section = ""

            text_content = hit.get("text", "").strip()
            tag = f"[Source-{idx}]"

            if source_url:
                source_map[tag] = {
                    "source_url": source_url,
                    "document_name": document_name,
                    "section": section,
                }

            context.append(f"{tag}\n{text_content}")

        formatted_context = "\n\n---\n\n".join(context)
        final_prompt = prompt_template.format(
            context=formatted_context,
            query=query,
            user_prompt=query,
        )

        # 4. Stream response from Groq
        chat_completion_stream = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.1,
            stream=True,
        )

        full_generated_text = ""

        async for chunk in chat_completion_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    full_generated_text += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        cited_tags = set(re.findall(r"\[Source-\d+\]", full_generated_text))
        used_sources = []
        seen_urls = set()

        target_tags = sorted(cited_tags) if cited_tags else list(source_map.keys())

        for tag in target_tags:
            if tag in source_map:
                meta = source_map[tag]
                url = meta["source_url"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    used_sources.append({
                        "id": tag,
                        "source_url": url,
                        "document_name": meta["document_name"],
                        "section": meta["section"],
                    })

        yield f"data: {json.dumps({'type': 'sources', 'sources': used_sources})}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        logger.info("Client disconnected during stream generation.")
        raise

    except Exception as e:
        logger.error(f"Error in generate_rag_response: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"