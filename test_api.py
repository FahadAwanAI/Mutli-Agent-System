import google.generativeai as genai

genai.configure(api_key="AIzaSyCYphwoDsIx4Z2E5RttOJeXmvI784ckVxo")
models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
