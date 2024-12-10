from flask import Flask, render_template, request, jsonify
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset (make sure your data.json contains relevant questions about RNN)
with open('data.json', 'r') as f:
    dataset = json.load(f)

# Load BERT model and tokenizer
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to find the most relevant answer
def find_answer(query, dataset):
    query_embedding = get_bert_embeddings(query)
    max_similarity = 0
    best_answer = ""
    
    for item in dataset:
        question = item['question']
        answer = item['answer']
        question_embedding = get_bert_embeddings(question)
        
        # Calculate cosine similarity between query and question embeddings
        similarity = cosine_similarity(query_embedding.numpy(), question_embedding.numpy())[0][0]
        
        # Debugging print to see the similarity scores and which questions are being matched
        print(f"Query: {query} - Question: {question} - Similarity: {similarity}")
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = answer
    
    return best_answer, max_similarity

# Home route to render the frontend
@app.route('/')
def index():
    return render_template('frontend.html')

# API route to process the user's question
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()  # Get JSON data from the frontend (contains the query)
        
        # Check if 'query' is in the data
        if 'query' not in data:
            return jsonify({'error': 'Query not provided'}), 400

        query = data['query']  # Extract the query from the data
        answer, similarity = find_answer(query, dataset)  # Get the answer using your existing function
        
        # Convert similarity to a standard Python float
        response = {
            'answer': answer,
            'accuracy': float(similarity)  # Convert to float here
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
