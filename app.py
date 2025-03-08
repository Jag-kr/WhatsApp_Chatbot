import os
from flask import Flask, request, render_template, abort
from twilio.twiml.messaging_response import MessagingResponse
from qa_processor import QAProcessor
from config import DOCUMENT_PATH, MAX_WHATSAPP_MESSAGE_LENGTH
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Initialize QA Processor
qa_processor = QAProcessor(DOCUMENT_PATH)

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    """Handle incoming WhatsApp messages."""
    try:
        # Verify it's a POST request
        if request.method != 'POST':
            abort(405)  # Method Not Allowed

        incoming_msg = request.values.get('Body', '').lower()

        # Create Twilio response object
        resp = MessagingResponse()

        # Handle greeting
        if incoming_msg == "hi":
            response = "Hello! I'm your FindMySchool assistant. How can I help you today?"
        else:
            # Get response from QA processor
            answer, confidence = qa_processor.get_response(incoming_msg)

            # Format response based on confidence
            if confidence < 0.1:
                response = "I'm sorry, but I couldn't find a relevant answer to your question. Could you please rephrase it?"
            else:
                response = answer

        # Truncate response if needed
        if len(response) > MAX_WHATSAPP_MESSAGE_LENGTH:
            response = response[:MAX_WHATSAPP_MESSAGE_LENGTH - 3] + "..."

        resp.message(response)
        return str(resp)

    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {str(e)}")
        resp = MessagingResponse()
        resp.message("I apologize, but I encountered an error processing your message.")
        return str(resp)

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors."""
    return 'Method Not Allowed', 405

if __name__ == "__main__":
    app.run(debug=True)