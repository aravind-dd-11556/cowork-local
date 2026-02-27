from __future__ import annotations
import streamlit as st
import asyncio
from io import StringIO
from typing import List


from typing import Literal, TypedDict
import asyncio
import os
from crawler_ai_docs import process_and_store_document, get_catalyst_urls, crawl_parallel
from crawler_ai_docs import chunk_text, process_chunk, insert_chunk

import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
# Streamlit App Configuration

# Import all the message part classes

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
st.set_page_config(page_title="AI Documentation Processor", layout="wide")

# App Title
st.title("AI Documentation Processor")
st.sidebar.header("Options")

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

# Tabs for Navigation
tabs = st.tabs(["Document Processing", "URL Crawling"])

# ---- Document Processing ----
with tabs[0]:
    st.header("Document Processing")
    uploaded_file = st.file_uploader("Upload a Text File", type=["txt", "md", "json"])
    input_text = st.text_area("Or Paste Text Below", height=200)
    process_button = st.button("Process Document")

    if process_button:
        if uploaded_file:
            # Read uploaded file
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif input_text:
            file_content = input_text
        else:
            st.error("Please upload a file or provide input text to process.")
            st.stop()

        # Process the document
        with st.spinner("Processing document..."):
            markdown = file_content
            url = "dummy-url-for-testing"  # Placeholder for input text
            asyncio.run(process_and_store_document(url, markdown))

        st.success("Document processed successfully!")

# ---- URL Crawling ----
with tabs[1]:
    st.header("URL Crawling")
    user_urls = st.text_area("Enter URLs (one per line)", height=200)
    crawl_button = st.button("Crawl URLs")

    if crawl_button:
        if user_urls.strip():
            urls = [url.strip() for url in user_urls.split("\n") if url.strip()]
            st.write(f"Found {len(urls)} URLs to crawl.")

            with st.spinner("Crawling URLs..."):
                asyncio.run(crawl_parallel(urls, max_concurrent=5))
            st.success("URL crawling complete!")
        else:
            st.error("Please provide at least one URL.")

# ---- Display Results ----
st.header("Processed Results")
if st.button("Show Results"):
    st.write("Displaying processed chunks (from Supabase)...")
    # Here you could fetch results from Supabase or local storage.
    st.write("Integration pending: Show results from the database.")

