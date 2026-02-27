from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os


from pydantic_ai import Agent, ModelRetry, RunContext
# from pydantic_ai.models.openai import GeminiModel
from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List


model = GeminiModel('gemini-1.5-flash', api_key='AIzaSyBWhZI4_YgDd6EZEj99C86GYfMdNhVOFR0')
# agent = Agent(model)

async_openai_client = AsyncOpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)



logfire.configure(send_to_logfire='if-token-present')

@dataclass
class CatalystAiDeps:
    supabase: Client
    o_client: AsyncOpenAI

system_prompt = """

'You are an expert at zcatalyst cli - a command line interface created for zoho catalyst. You have access to all its documentations, including examples, command reference, and other supporting resources to generate steps for achieving something in catalyst with the CLI commands and debug any issues in it.'
        'Use the `retrieve_relevant_documentation` tool to get the relevant documentation chunks based on the user query'
        # 'Optionally use the `list_documentation_pages` tool to get all the available urls of the documentation.'
        # 'Then use `get_page_content` tool to retrieve the full content of a specific documentation page url by combining all its chunks.'
        'When you first look at the documentation, always start with RAG.'
        # 'Then also always check the list of available documentation pages and retrieve the content of page(s) if it\'ll help.'
        # 'IMPORTANT: Give the link to the relavent documentation url whenever possible.'
        'Always let the user know when you didn\'t find the answer in the documentation or the right URL - be honest.'
"""

catalyst_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=CatalystAiDeps,
    retries=2
)

async def get_embedding(text: str,o_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        print('get embedding called')
        response = await o_client.embeddings.create(
            input=text,
            model="nvidia/nv-embed-v1", #nvidia/nv-embed-v1
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"}
        )
        embedding= response.data[0].embedding
        print("embedding data: " + str(embedding))
        return embedding
        # embedding = adjust_embedding(embedd, target_dim=1536)
        # return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@catalyst_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[CatalystAiDeps], user_query: str) -> str:
    """
        Retrieve relevant documentation chunks based on the query with RAG.

        Args:
            ctx: The context including the Supabase client and Google client
            user_query: The user's question or query

        Returns:
            A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        print('retrieve_relevant_documentation called')
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.o_client)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'catalyst_ai_docs'}
            }
        ).execute()

        if not result.data:
            return "No Relevant Documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

            

            # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@catalyst_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[CatalystAiDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'catalyst_ai_docs') \
            .execute()

        if not result.data:
            return []

        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@catalyst_ai_expert.tool
async def get_page_content(ctx: RunContext[CatalystAiDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'catalyst_ai_docs') \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"