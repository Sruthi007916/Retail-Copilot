import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def ask_gemini(question, context):
    prompt = f"""
You are a Retail Sales and Inventory Copilot.

Your job is to help a retail manager make decisions using the provided
sales and inventory data.

IMPORTANT RULES:
1. Use ONLY the data provided below.
2. Never invent or guess numbers.
3. Always use the exact product names and numbers from the data.
4. If the data is not sufficient to answer the question, clearly say:
   "The available data is not sufficient to answer this question."
5. Give a practical recommendation when appropriate.
6. Keep the answer concise and easy to understand.
7. When giving a recommendation, explain which data supports it.
8. Do not claim information that is not present in the data.

AVAILABLE RETAIL DATA:
{context}

MANAGER QUESTION:
{question}

Answer the manager clearly using the available data.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text