# Chatbot

A conversational AI chatbot built using LangChain and OpenAI's llms

## Features

* It uses LangChain methods to load documents and splits them with a specified size and overlap. 
* Each chunk is created into an embedding using OpenAI's Embeddings and stored in Chroma, a vector database. 
* Vector Db is capable of doing semantic search or MMR on an input question.
* Various LangChain's chain are provided to allow efficient retrieval with prompts and contextual memory 

## Requirements

* Python 3.x
* OpenAI's API Key: To get your OpenAI API key, visit https://platform.openai.com/account/api-keys

## Usage

1. Clone the repository: `git clone https://github.com/arushivashist/ai-chatbot.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the chatbot using steps defined in `chatbot.py`

## Example Use Cases

* [Provide examples of how to interact with your chatbot, e.g., user input and expected responses]