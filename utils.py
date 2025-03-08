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
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        return []

def load_document(file_path: str) -> str:
    """
    Load document from file.
    
    Args:
        file_path (str): Path to the document
        
    Returns:
        str: Document content
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        return ""
