import os
import logging
from flask import Flask, request, render_template, abort
from twilio.twiml.messaging_response import MessagingResponse
from qa_processor import QAProcessor
from config import DOCUMENT_PATH, MAX_WHATSAPP_MESSAGE_LENGTH
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv(
    "SESSION_SECRET", "default_secret_key"
)  # Fallback for safety

# Initialize QA Processor
qa_processor = QAProcessor(DOCUMENT_PATH)


def process_whatsapp_message(message):
    """Process incoming WhatsApp messages and return a response."""
    message = message.lower().strip()  # Normalize input

    if not message:
        return "Please send a valid query."

    if message in ["hi", "hello"]:
        return "Hello! I'm your FindMySchool assistant. How can I help you today?"

    # Get response from QA processor
    answer, confidence = qa_processor.get_response(message)

    # Format response
    if confidence < 0.1:
        return "I'm sorry, but I couldn't find a relevant answer. Could you please rephrase it?"

    return (
        answer[: MAX_WHATSAPP_MESSAGE_LENGTH - 3] + "..."
        if len(answer) > MAX_WHATSAPP_MESSAGE_LENGTH
        else answer
    )


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    """Handle incoming WhatsApp messages."""
    try:
        incoming_msg = request.values.get("Body", "")
        response_text = process_whatsapp_message(incoming_msg)

        # Send response via Twilio
        resp = MessagingResponse()
        resp.message(response_text)
        return str(resp)

    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}")
        resp = MessagingResponse()
        resp.message("I apologize, but I encountered an error processing your message.")
        return str(resp)


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors."""
    return "Method Not Allowed", 405


if __name__ == "__main__":
    app.run()
