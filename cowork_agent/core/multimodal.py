"""
Multi-Modal Input — Support for image content in messages.

Provides utilities to:
  1. Detect image file paths in user messages
  2. Encode images to base64 content blocks
  3. Build mixed text+image message payloads
  4. Convert between provider-specific image formats

Supported formats: PNG, JPEG, GIF, WebP
Max image size: 20MB (Anthropic limit)

Sprint 4 (P2-Advanced) Feature 4.
"""

from __future__ import annotations
import base64
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported image MIME types
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Maximum image file size (20MB — Anthropic's limit)
MAX_IMAGE_SIZE = 20 * 1024 * 1024


@dataclass
class ImageContent:
    """A single image content block."""
    media_type: str       # e.g. "image/png"
    base64_data: str      # base64-encoded image bytes
    source_path: str = "" # original file path (for logging)
    size_bytes: int = 0

    def to_anthropic_block(self) -> dict:
        """Convert to Anthropic's image content block format."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.base64_data,
            },
        }

    def to_openai_block(self) -> dict:
        """Convert to OpenAI's image content block format."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.media_type};base64,{self.base64_data}",
            },
        }


@dataclass
class MultiModalMessage:
    """
    A message that can contain both text and image content blocks.
    Used as an intermediate representation before provider-specific conversion.
    """
    text: str = ""
    images: list[ImageContent] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0

    def to_anthropic_content(self) -> list[dict]:
        """Convert to Anthropic's content block array."""
        blocks = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for img in self.images:
            blocks.append(img.to_anthropic_block())
        return blocks

    def to_openai_content(self) -> list[dict]:
        """Convert to OpenAI's content block array."""
        blocks = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for img in self.images:
            blocks.append(img.to_openai_block())
        return blocks


def load_image(file_path: str) -> Optional[ImageContent]:
    """
    Load an image file and return an ImageContent object.

    Returns None if:
      - File doesn't exist
      - File is not a supported image type
      - File exceeds size limit
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"Image file not found: {file_path}")
        return None

    # Check extension
    ext = path.suffix.lower()
    media_type = SUPPORTED_IMAGE_TYPES.get(ext)
    if not media_type:
        logger.warning(f"Unsupported image type: {ext} (file: {file_path})")
        return None

    # Check size
    size = path.stat().st_size
    if size > MAX_IMAGE_SIZE:
        logger.warning(
            f"Image too large: {size / 1024 / 1024:.1f}MB > "
            f"{MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB limit (file: {file_path})"
        )
        return None

    if size == 0:
        logger.warning(f"Image file is empty: {file_path}")
        return None

    # Read and encode
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return ImageContent(
            media_type=media_type,
            base64_data=b64,
            source_path=str(path),
            size_bytes=size,
        )
    except Exception as e:
        logger.error(f"Failed to read image {file_path}: {e}")
        return None


def extract_image_paths(text: str) -> list[str]:
    """
    Extract file paths from text that look like image references.

    Looks for:
      - Absolute paths ending in supported extensions
      - Relative paths ending in supported extensions
    """
    import re

    paths = []
    # Match paths (absolute or relative) ending in image extensions
    pattern = r'(?:^|\s)((?:/|\.{1,2}/)[^\s]+\.(?:png|jpg|jpeg|gif|webp))'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        candidate = match.group(1).strip()
        if os.path.exists(candidate):
            paths.append(candidate)

    return paths


def parse_multimodal_input(text: str, image_paths: Optional[list[str]] = None) -> MultiModalMessage:
    """
    Parse user input into a MultiModalMessage.

    If image_paths is provided, use those directly.
    Otherwise, try to extract image paths from the text.
    """
    if image_paths is None:
        image_paths = extract_image_paths(text)

    images = []
    for path in image_paths:
        img = load_image(path)
        if img:
            images.append(img)
            logger.info(f"Loaded image: {path} ({img.size_bytes / 1024:.1f}KB)")

    return MultiModalMessage(text=text, images=images)
