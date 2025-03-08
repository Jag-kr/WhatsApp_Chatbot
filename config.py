import os
import logging

# Application Configuration
CHUNK_SIZE = 1000
MAX_WHATSAPP_MESSAGE_LENGTH = 1600
DOCUMENT_PATH = "attached_assets/FindMySchool Data.txt"

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
