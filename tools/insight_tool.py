from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_insights(df):

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    sample = df.head(20).to_string()

    prompt = f"""
    You are a senior data scientist.

    Analyze the dataset sample below and explain:

    1. Important patterns
    2. Potential trends
    3. Possible relationships
    4. Data quality issues

    Dataset sample:

    {sample}
    """

    response = llm.invoke(prompt)

    return response.content