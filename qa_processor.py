import logging
from typing import Tuple, List, Dict, Optional
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import CHUNK_SIZE
from utils import chunk_text, load_document

logger = logging.getLogger(__name__)

# Try to import sentence transformers for better semantic matching
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class QAProcessor:
    def __init__(
        self,
        document_path: str,
        use_sentence_transformers: bool = False,
        low_confidence_threshold: float = 0.1,
        high_confidence_threshold: float = 0.3,
        max_chunks: int = 3,
    ):
        """
        Initialize QA processor with document path.

        Args:
            document_path (str): Path to the document file
            use_sentence_transformers (bool): Whether to use sentence transformers for embeddings
            low_confidence_threshold (float): Threshold below which answers are rejected
            high_confidence_threshold (float): Threshold above which single answers are returned
            max_chunks (int): Maximum number of chunks to include in medium confidence responses
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.max_chunks = max_chunks

        # Set up vectorizer based on availability and preference
        self.use_sentence_transformers = (
            use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
        )

        if self.use_sentence_transformers:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.vectorizer = None
        else:
            self.vectorizer = TfidfVectorizer(
                min_df=1,
                stop_words="english",
                ngram_range=(1, 3),  # Include trigrams for better matching
            )
            self.model = None

        self.texts = []
        self.doc_vectors = None
        self.initialize_processor(document_path)

    def _create_chunks(self, text: str) -> List[str]:
        """Create semantically meaningful chunks from text"""
        # First try to split by double newlines (paragraphs)
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        # Process very large chunks further using the chunk_text utility
        result_chunks = []
        for chunk in chunks:
            if len(chunk) > CHUNK_SIZE:
                result_chunks.extend(chunk_text(chunk, CHUNK_SIZE))
            else:
                result_chunks.append(chunk)

        return result_chunks

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

            # Create semantically meaningful chunks
            self.texts = self._create_chunks(text)

            if not self.texts:
                raise ValueError("No text chunks created")

            # Create vector representations
            if self.use_sentence_transformers:
                self.doc_vectors = self.model.encode(self.texts)
            else:
                self.doc_vectors = self.vectorizer.fit_transform(self.texts)

            logger.info(f"QA Processor initialized with {len(self.texts)} chunks")
        except Exception as e:
            logger.error(f"Error initializing QA processor: {str(e)}")
            self.texts = [
                "I apologize, but I'm currently unable to access the information. Please try again later."
            ]
            if self.use_sentence_transformers:
                self.doc_vectors = self.model.encode(self.texts)
            else:
                self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    @lru_cache(maxsize=128)
    def get_response(self, query: str) -> Tuple[str, float]:
        """
        Get response for a query using similarity search.
        Results are cached for better performance.

        Args:
            query (str): User query

        Returns:
            Tuple[str, float]: Response text and similarity score
        """
        try:
            if self.use_sentence_transformers:
                query_vector = self.model.encode([query])
                similarities = cosine_similarity([query_vector], self.doc_vectors)[0]
            else:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

            # Get top matches
            top_indices = similarities.argsort()[-self.max_chunks :][::-1]
            best_match_idx = top_indices[0]
            similarity_score = similarities[best_match_idx]

            # If similarity is very low, return the default message
            if similarity_score < self.low_confidence_threshold:
                return (
                    "I'm sorry, but I couldn't find a relevant answer to your question. Could you please rephrase it?",
                    0.0,
                )

            # If we have a high confidence match, return the most relevant chunk
            if similarity_score >= self.high_confidence_threshold:
                return self.texts[best_match_idx], similarity_score

            # For medium confidence, combine relevant chunks
            relevant_indices = [
                idx
                for idx in top_indices
                if similarities[idx] > self.low_confidence_threshold
            ]
            response = "\n\n".join([self.texts[idx] for idx in relevant_indices])
            return response, similarity_score

        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return "I apologize, but I couldn't process your query at this time.", 0.0
