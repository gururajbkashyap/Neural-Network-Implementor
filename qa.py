import json
import torch
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import faiss
import numpy as np

# Load dataset
with open('data.json', 'r') as f:
    dataset = json.load(f)

# Load BERT model and tokenizer
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Load fine-tuned GPT-2 model and tokenizer
gpt2_model_name = "./finetuned_model"  # Path to the fine-tuned GPT-2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()  # Return as NumPy array for FAISS

# Precompute embeddings for all questions in the dataset
question_embeddings = []
questions = []
for item in dataset:
    question = item['question']
    question_embeddings.append(get_bert_embeddings(question))
    questions.append(question)

# Convert question embeddings to unit vectors for cosine similarity
question_embeddings = np.vstack(question_embeddings).astype('float32')
question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

# Build a FAISS index
index = faiss.IndexFlatIP(question_embeddings.shape[1])  # Using Inner Product for cosine similarity
index.add(question_embeddings)

# GPT-2 response generation with fine-tuned model
def generate_gpt2_response(query):
    inputs = gpt2_tokenizer.encode(query, return_tensors='pt', truncation=True, max_length=100)
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to find answer with fallback to fine-tuned GPT-2
def find_answer_with_fallback(query, dataset, threshold=0.85):
    query_embedding = get_bert_embeddings(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize the query embedding
    D, I = index.search(query_embedding, k=1)  # Search for the closest match
    best_match_index = I[0][0]
    similarity = D[0][0]  # Cosine similarity (already normalized in FAISS)
    
    if similarity >= threshold:
        # High similarity - return the matched dataset answer
        best_answer = dataset[best_match_index]['answer']
        return best_answer, similarity
    else:
        # Low similarity - fallback to GPT-2 for generating a response
        gpt2_response = generate_gpt2_response(query)
        return gpt2_response, similarity

# Main interaction loop
print("Ask me anything from your dataset or GPT-2 will assist with queries beyond the dataset!")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    answer, similarity = find_answer_with_fallback(user_query, dataset)
    if similarity >= 0.85:
        print(f"Bot: {answer}")
        print(f"Similarity Score: {similarity:.2f}")
    else:
        print(f"Bot (via GPT-2): {answer}")
        print("Bot: I used GPT-2 to answer this query as no relevant dataset match was found.")
