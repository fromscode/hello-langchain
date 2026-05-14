from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

template_1 = """
    You are a product designer heavily influenced by Steve Jobs with over 30 years of experience in customer facing roles.

    Generate three brand names for a new shoe company that produces shoes that can fit any size.
    The output should be a single string with three brand names separated by a commma and no other punctuations.

    Eg: brand-name-1, brand-name-2, brand-name-3
"""

template_2 = """
    You are a market penetration analyst with 25+ years of experience in predicting which brand names can penetrate the market the most.
    
    Analyze the three brand names for a shoe company that produces shoes that can fit any shoe size: {brand_names}.
    Score the brand names on a scale of 1-5 (1 being the lowest and 5 being the highest) on two categories: Catchiness, Classy

    The output should only be a json in the following format:
    [{{
    "brand-name": brand-name-1,
    "catchiness": 3,
    "classy": 2
    }},
    {{
    "brand-name": brand-name-2,
    "catchiness": 2,
    "classy": 3
    }},
    {{
    "brand-name": brand-name-3,
    "catchiness": 1,
    "classy": 2
    }}]
"""

prompt_1 = PromptTemplate(
    template=template_1
)

prompt_2 = PromptTemplate(input_variables=["brand_names"], template=template_2)

product_designer = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

market_analyst = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

def postProcessing(output):
    brands = []
    for brand in output:
        penetration_ability = brand["catchiness"] * 10 + brand["classy"] * 10
        brands.append([brand["brand-name"], penetration_ability])

    brands.sort(key=lambda a: a[1], reverse=True)
    
    for brand in brands:
        print(f"Brand Name: {brand[0].capitalize()}. Success percentage: {brand[1]}")

    # print(output)

chain = (
    prompt_1
    | product_designer
    | StrOutputParser()
    | (lambda output: {"brand_names" : output} )
    | prompt_2
    | market_analyst
    | JsonOutputParser()
    | postProcessing
)

chain.invoke({})