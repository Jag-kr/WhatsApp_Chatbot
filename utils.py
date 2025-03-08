import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks of specified size.

    Args:
        text (str): Input text to be chunked
        chunk_size (int): Size of each chunk

    Returns:
        List[str]: List of text chunks
    """
    try:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        return []


def load_document(file_path: str) -> str:
    """
    Load document from file using os checks.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""

        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return ""

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Document Loaded: {content[:500]}")  # Print first 500 characters
            return content
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        return ""
