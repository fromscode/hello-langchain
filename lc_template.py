from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

template = """
    You are a senior site reliability engineer.
    Analyze the following server log from the {service_name} microservice.

    Log entry:
    {log_data}

    Identify the root cause of the error. Keep your explaination under 3 sentences.
"""

prompt = PromptTemplate(
    input_variables=["service_name", "log_data"],
    template=template
)

chain = prompt | llm

print("Analyzing log...")

response = chain.invoke({
    "service_name": "PaymentGateway",
    "log_data": "[ERROR] 2026-05-14 10:14:26 - 502 Bad Gateway. Connection closed."
})

print("\n-- SRE DIAGNOSIS --")
print(response.text)