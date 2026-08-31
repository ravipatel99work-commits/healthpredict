import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# list all models your API key can access
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
