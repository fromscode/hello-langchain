from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

""" The Goal here is to chain one prompt to another

1. Prompt-> Customer complaint, response -> Identify which microservice is failing
2. Based on that microservice, generate a 3 step debugging process for a junior engineer
 """

template_1 = """
You are a senior Site Reliability Enginner with an 25+ years of experience in understanding user complaints and correctly mapping them to the 
faulty microservice.

Based on the user's complaint, find the microservice that is actually failing.
The complete list of microservices: 'AuthService', 'BookingService', 'PaymentService', 'OrderTrackingService'

The output should be just the microservice that is most likely to be failing and nothing else

User's complaint: {complaint}
"""

template_2 = """"
You are a senior Site Reliability Enginner with an 25+ years of experience in debugging systems and services.

Recently it was found that the ${service} is failing / down.

Generate a 3 step debug process to get the service back online. Generate only 3 debugging steps and nothing else.
"""

prompt_1 = PromptTemplate(
    input_variables=["complaint"],
    template=template_1
)

prompt_2 = PromptTemplate(
    input_variables=["service"],
    template=template_2
)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

chain = (
    prompt_1
    | llm
    | StrOutputParser()
    | (lambda output : {"service" : output})
    | prompt_2
    | llm
    |StrOutputParser()
)

print("Executing")

complaint = "Money is getting deducted from my bank account, but I am not getting any confirmation from the app"

final_result = chain.invoke({"complaint": complaint})

print(final_result)