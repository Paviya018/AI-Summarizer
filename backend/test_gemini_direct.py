import os
from google import genai

print("1. Starting program...")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set")
    exit()

print("2. API key found")

client = genai.Client(
    api_key=api_key,
    http_options={
        "timeout": 30000
    }
)

print("3. Client created")
print("4. Sending request...")

try:
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input="Say hello in one sentence.",
        generation_config={
            "thinking_level": "low"
        }
    )

    print("5. Response received!")
    print()
    print(response.output_text)
    print()
    print("6. Test completed.")

except Exception as e:
    print("\n===== ERROR =====")
    print("Error type:", type(e).__name__)
    print("Error:", e)
    print("================")