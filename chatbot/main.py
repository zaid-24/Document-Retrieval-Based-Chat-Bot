from fastapi import FastAPI, HTTPException
import openai
from bs4 import BeautifulSoup
import requests

app = FastAPI()

# Set your OpenAI API key
openai.api_key = ""

# Function to search a website for the answer using BeautifulSoup
def search_website(query, url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Implement your logic to extract relevant information from the website
    # For simplicity, let's assume the answer is in the text of the paragraphs
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        if query.lower() in paragraph.text.lower():
            return paragraph.text
    return None

# Function to get an answer from ChatGPT
def chat_gpt(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model for your use case
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            max_tokens=2000
        )
        if response is None or response.get("choices") is None:
            raise HTTPException(status_code=500, detail="OpenAI response is empty")
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error with ChatGPT: {e}")
        return None

# API endpoint to answer questions
@app.get("/answer/{question}")
def get_answer(question: str):
    # Define the websites to search
    websites = ["https://www.ready-able.com/10-common-plumbing-repair-questions-answered/"]

    # Search each website for the answer
    for website in websites:
        website_answer = search_website(question, website)
        if website_answer:
            return {"source": website, "answer": website_answer}

    # If the answer is not found in the websites, ask ChatGPT
    gpt_answer = chat_gpt(question)
    return {"source": "ChatGPT", "answer": gpt_answer}
