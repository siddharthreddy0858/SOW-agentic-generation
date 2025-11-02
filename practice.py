from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.5
)

# Parser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# Prompts
prompt_1 = "Give me a list of 5 countries in {continent}, {format_instructions}"
countries_prompt = PromptTemplate(template = prompt_1,input_variables=['continent'],partial_variables={'format_instructions':format_instructions}) | llm | output_parser

prompt_2 = "For the following countries: {input}, give me their capitals in a comma-separated list."
capitals_prompt = PromptTemplate.from_template(prompt_2) | llm | output_parser

# ✅ Single pipeline one-liner
full_pipeline = countries_prompt | capitals_prompt

# Run
result = full_pipeline.invoke({"continent": "Asia"})
print("Capitals:", result)
