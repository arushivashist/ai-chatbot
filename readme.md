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

* python3 chatbot.py -q "What's the agenda of this course ?"
    > The agenda of this course, CS229, is to teach machine learning. The instructor, Andrew Ng, aims to convey excitement about machine learning, enable students to apply state-of-the-art machine learning algorithms to various problems, and prepare them for research in machine learning. The course covers topics like understanding the human genome, potentially contributing to finding a cure for cancer, and practical applications like building systems for reading zip codes. The course logistics include prerequisites of basic computer science knowledge and skills, online resources like lecture notes and a newsgroup for discussions, and contact information for the teaching staff.
* python3 chatbot.py -q "What are major topics for this class?"
    > The major topics for this class include machine learning, extensions of machine learning concepts, online resources for the course, programming in MATLAB or Octave, basic probability and statistics, and basic linear algebra.
* python3 chatbot.py -q "Is probability a class topic?"
    > Yes, probability is assumed to be a prerequisite for the class, as mentioned in the context provided. The instructor assumes familiarity with basic probability and statistics, so it is not explicitly mentioned as a topic that will be covered in the class.
* python3 chatbot.py -q "why are those prerequesites needed?"
    > The prerequisites mentioned in the context, such as basic knowledge of computer science, probability and statistics, and basic linear algebra, are needed for the machine learning class to ensure that students have the foundational understanding required to grasp the concepts and techniques taught in the course. These prerequisites help students apply state-of-the-art machine learning algorithms effectively to various problems and prepare them for potential research in machine learning.