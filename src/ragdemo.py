import ollama
import numpy as np

# Load dataset
dataset = []
with open('src/cat-facts.txt', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

# Define models
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


# Retrival Setup

# Create in memory vector database
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

# Populate vector database
for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to vector database')

# Function to compute cosine similarity utilizig numpy
def cosine_similarity_np(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity_np(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
    

# Chatbot Setup

input_query = input("Ask a question: ")
retrieved_knoledge = retrieve(input_query)

print("Retrieved Knowledge:")
for chunk, similarity in retrieved_knoledge:
    print(f" = (similarity: {similarity:.2f}) {chunk}")

instruction_prompt = f''' You are a helpful chatbot. Use only the following pieces of 
    context to answer the question. Don't make up any new imformation:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knoledge])}
    '''

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": input_query}
    ],
    stream=True

)

print("Chatbot Response:")
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
