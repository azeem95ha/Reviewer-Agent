import os
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from classes import ComparisonFeatures
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

model_with_structured_output = model.with_structured_output(ComparisonFeatures)

results = model_with_structured_output.invoke(" features:" \
"sprinkler type 1, sprinkler type 2, K-Factor, Temperature rating, Activation Mechanism, Response Time, Compliance Standards")

print(str(results.features))