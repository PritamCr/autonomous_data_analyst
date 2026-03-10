from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import re, json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def suggest_features(df, target):

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    columns = df.columns.tolist()

    prompt = f"""
    You are an expert ML engineer.
    Dataset columns: {columns}
    Target variable: {target}

    Suggest the best features to train the model
    for maximum accuracy. Return a JSON array of column names ONLY.
    Example output: ["col1", "col2", "col3"]
    """

    response = llm.invoke(prompt)
    text = response.content

    # 1️⃣ Remove ```json``` code blocks if present
    text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)

    # 2️⃣ Extract JSON array from text
    try:
        features = json.loads(text)
    except:
        # fallback: extract words between quotes
        features = re.findall(r'"(.*?)"', text)

    # 3️⃣ Remove the target if it accidentally appears
    features = [f for f in features if f != target]

    return features