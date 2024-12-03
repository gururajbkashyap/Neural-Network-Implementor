import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
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
    # Average the token embeddings for sentence embedding
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
        
        # Compute similarity
        similarity = cosine_similarity(query_embedding.numpy(), question_embedding.numpy())[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = answer
    
    return best_answer, max_similarity

# Main interaction loop
print("Ask me anything from your dataset!")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # Directly find and display the answer
    answer, accuracy = find_answer(user_query, dataset)
    print(f"Bot: {answer}")
    print(f"Accuracy: {accuracy:.2f}")

    # Provide additional prompt for low confidence
    if accuracy < 0.70:
        print("Bot: My confidence is low for this answer. Could you clarify or rephrase your question?")
