import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import CHUNK_SIZE
from utils import chunk_text, load_document

logger = logging.getLogger(__name__)

class QAProcessor:
    def __init__(self, document_path: str):
        """
        Initialize QA processor with document path.

        Args:
            document_path (str): Path to the document file
        """
        self.vectorizer = TfidfVectorizer(
            min_df=1,  # Include terms that appear in at least 1 document
            stop_words='english'  # Remove common English stop words
        )
        self.texts = []
        self.doc_vectors = None
        self.initialize_processor(document_path)

    def initialize_processor(self, document_path: str) -> None:
        """
        Initialize the processor by loading and processing the document.

        Args:
            document_path (str): Path to the document file
        """
        try:
            text = load_document(document_path)
            if not text:
                raise ValueError("Document is empty")

            self.texts = chunk_text(text, CHUNK_SIZE)
            if not self.texts:
                raise ValueError("No text chunks created")

            self.doc_vectors = self.vectorizer.fit_transform(self.texts)
            logger.info("QA Processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA processor: {str(e)}")
            # Initialize with a default response if document loading fails
            self.texts = ["I apologize, but I'm currently unable to access the school information. Please try again later."]
            self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def get_response(self, query: str) -> Tuple[str, float]:
        """
        Get response for a query using TF-IDF similarity.

        Args:
            query (str): User query

        Returns:
            Tuple[str, float]: Response text and similarity score
        """
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.doc_vectors)
            best_match_idx = similarities.argmax()
            similarity_score = similarities[0][best_match_idx]

            return self.texts[best_match_idx], similarity_score
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I apologize, but I couldn't process your query at this time.", 0.0