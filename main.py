from __future__ import annotations

import streamlit as st
import asyncio
from io import StringIO
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from typing import Literal, TypedDict
import asyncio
import os
from crawler_ai_docs import process_and_store_document, get_catalyst_urls, crawl_parallel
from crawler_ai_docs import chunk_text, process_chunk, insert_chunk

import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
# Streamlit App Configuration

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from catalyst_ai_expert import catalyst_ai_agent,PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# model = AsyncOpenAI(
#   api_key="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",
#   base_url="https://integrate.api.nvidia.com/v1"
# )


model = GeminiModel('gemini-1.5-flash', api_key='AIzaSyBWhZI4_YgDd6EZEj99C86GYfMdNhVOFR0')
agent = Agent(model)

supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=model
    )

    # Run the agent in a stream
    async with catalyst_ai_agent.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Catalyst AI Agentic RAG")
    st.write("Ask any question about Catalyst AI, the hidden truths of the beauty of this framework lie within.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about catalyst AI?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


# def main_sync():
#     asyncio.run(main())
if __name__ == "__main__":
    st.title("Catalyst AI Experts")
    asyncio.run(main())