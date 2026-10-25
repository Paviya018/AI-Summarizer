from services.summarizer_service import summarize_text

print("Starting Gemini test...")

text = """
Artificial intelligence is a field of computer science.
Machine learning allows computers to learn patterns from data.
"""

print("Calling Gemini...")

try:
    result = summarize_text(text)

    print("\n===== RESULT =====")
    print(result)
    print("==================")

except Exception as e:
    print("\n===== ERROR =====")
    print(type(e).__name__)
    print(e)
    print("================")