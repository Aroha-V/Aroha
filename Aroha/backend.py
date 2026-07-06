import os
import time
import logging
from flask import Flask, send_from_directory, request, jsonify
from return_context import return_context
from google import genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ROOT = os.path.dirname(__file__)
app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any static asset (logo.png, etc.) from the project root."""
    return send_from_directory(ROOT, filename)


@app.route("/Chatbot", methods=["POST"])
def chatbot():
    user_input = request.form.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    log.info("User query: %s", user_input)

    context = return_context(user_input)

    if context == "query_error":
        return jsonify(
            "I couldn't find any relevant outbreak records for your query. "
            "Try specifying a state, district, or disease name (e.g. 'dengue in Tamil Nadu')."
        )

    prompt = f"""You are AROHA, an AI assistant for India's IDSP disease surveillance data.

Rules:
1. Answer ONLY using the provided context records — never use outside knowledge.
2. Never invent case numbers, dates, or locations.
3. If the user asks about a SPECIFIC time period (e.g. "March 2021") and no records match that period:
   - Still show the most relevant records from the context
   - Clearly note that records for that specific period are not available in the dataset
   - Do NOT say "No relevant data found" when relevant location/disease records DO exist
4. When multiple records exist for the same location/disease, summarise clearly.
5. Always cite state and district when giving figures.
6. Format numbers clearly (e.g. "1,234 cases").
7. Disease names in the dataset may have prefixes (e.g. "v. Dengue") — treat them as the same disease.

Context:
{context}

Question:
{user_input}

Answer:"""

    answer = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            answer = response.text.strip()
            break
        except Exception as e:
            log.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)   # 1s, 2s backoff
            else:
                log.error("Gemini all retries exhausted: %s", e)
                return jsonify("Gemini is overloaded right now — please try again in a moment."), 503

    log.info("Response length: %d chars", len(answer))
    return jsonify(answer)


if __name__ == "__main__":
    app.run(debug=True, port=2000)
