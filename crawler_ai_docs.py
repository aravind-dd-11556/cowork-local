import os
import sys
from http.client import responses

import psutil
import asyncio
import json
import requests
from typing import List, Dict, Any
from xml.etree import ElementTree
from urllib.parse import urlparse
from asyncio import Semaphore

from openai import AsyncOpenAI, OpenAI
from supabase import *
from dataclasses import dataclass
from datetime import datetime, timezone
__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

from idna import check_hyphen_ok
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)



client = AsyncOpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# response = client.embeddings.create(
#     input=["What is the capital of France?"],
#     model="nvidia/nv-embedqa-e5-v5",
#     encoding_format="float",
#     extra_body={"input_type": "query", "truncate": "NONE"}
# )


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks


# async def adjust_embedding(embedding: List[float], target_dim: int = 1536) -> List[float]:
#     if len(embedding) > target_dim:
#         return embedding[:target_dim]
#     elif len(embedding) < target_dim:
#         return embedding + [0] * (target_dim - len(embedding))
#     return embedding

def truncate_embedding(embedding: List[float], target_dim: int = 1536) -> List[float]:
    """Truncate the embedding to the desired dimension."""
    return embedding[:target_dim]

 # Create a semaphore to limit concurrent NVIDIA API calls
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""

    MAX_CONCURRENT = 3
    MAX_RETRIES = 3
    BACKOFF_TIME = 1

    semaphore = Semaphore(MAX_CONCURRENT)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                completion = await client.chat.completions.create(
                    model="qwen/qwen2.5-7b-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                    ],
                    stream=False
                )

                if hasattr(completion, '__dict__'):
                    completion_dict = completion.model_dump()
                else:
                    completion_dict = completion
                print(completion)
                print(completion_dict)
                content = completion_dict["choices"][0]["message"]["content"]
                content = content.strip()

                # Clean up the content
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:-3]
                elif content.startswith('```'):
                    content = content[3:-3]

                parsed_content = json.loads(content)
                return {
                    'title': parsed_content['title'].strip(),
                    'summary': parsed_content['summary'].strip()
                }

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_TIME * (2 ** attempt))
                    continue
                return {
                    "title": "Processing Error",
                    "summary": f"Failed after {MAX_RETRIES} attempts: {str(e)}"
                }


async def pad_embedding(embedding: List[float], target_dim: int = 1536) -> List[float]:
    """Pad the embedding to the desired dimension with zeros."""
    return embedding + [0] * (target_dim - len(embedding))


async def get_embedding(text: str,target_dim: int = 1536) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await client.embeddings.create(
            input=text,
            model="nvidia/nv-embed-v1", #nvidia/nv-embed-v1
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"}
        )
        embedding= response.data[0].embedding
        if len(embedding) > target_dim:
            embedding = truncate_embedding(embedding, target_dim=target_dim)
        elif len(embedding) < target_dim:
            embedding = pad_embedding(embedding, target_dim=target_dim)
        return embedding
        # embedding = adjust_embedding(embedd, target_dim=1536)
        # return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary

    await asyncio.sleep(0.5)
    extracted = await get_title_and_summary(chunk, url)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": "catalyst_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None
    


async def process_and_store_document(url: str,markdown:str):
    """Process a document and store it in chunks in parallel. """
    # Split into Chunks
    chunks = chunk_text(markdown)
    print("split started")
    # Process chunks in parallel

    tasks = [
        process_chunk(chunk,i,url)
        for i,chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    print("processed_chunks")
    # Store Chunks in parallel

    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def get_catalyst_urls():
    # url = "https://docs.catalyst.zoho.com/en/cli/v1/cli-command-reference/",
    async with AsyncWebCrawler() as crawler:
        try:
            resp = await crawler.arun(url="https://docs.catalyst.zoho.com/en/")
            internal_links = resp.links.get("internal", [])
            hrefs = [link.get("href") for link in internal_links if "href" in link]
            print(hrefs)
            # hrefs=["https://docs.catalyst.zoho.com/en/cli/v1/"]
            return hrefs

        except Exception as e:
            print(f"Error fetching url: {e}")
            return []


async def main():
    urls = await get_catalyst_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls, max_concurrent=10)
    else:
        print("No URLs found to crawl")


if __name__ == "__main__":
    asyncio.run(main())