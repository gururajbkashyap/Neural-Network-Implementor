import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load raw data
raw_data = """
Neural networks are machine learning models inspired by the human brain. They consist of layers of interconnected nodes, or neurons. Backpropagation is a key algorithm for training these networks by adjusting weights based on error gradients. Convolutional Neural Networks (CNNs) are a type of neural network effective for image processing tasks. Dropout is a regularization method to prevent overfitting by randomly deactivating neurons during training. Recurrent Neural Networks (RNNs) are designed for sequential data such as text or time series.
"""

# Split raw data into segments (paragraphs)
segments = raw_data.split("\n\n")  # Split by paragraphs or logical units
segments = [seg.strip() for seg in segments if seg.strip()]  # Clean empty segments

# Load SentenceTransformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient and accurate for semantic similarity

# Function to get embeddings
def get_embeddings(text):
    return st_model.encode(text, convert_to_tensor=True)

# Embed all segments
segment_embeddings = [get_embeddings(segment) for segment in segments]

# Function to find the most relevant segments for a query
def find_answer_from_raw_data(query, segments, segment_embeddings, top_k=3):
    query_embedding = get_embeddings(query)
    similarities = [
        (segment, cosine_similarity(query_embedding.unsqueeze(0).numpy(), embedding.unsqueeze(0).numpy())[0][0])
        for segment, embedding in zip(segments, segment_embeddings)
    ]
    # Sort by similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Retrieve top-k similar segments
    top_segments = [sim[0] for sim in similarities[:top_k]]
    max_similarity = similarities[0][1]  # Top similarity score
    best_answer = " ".join(top_segments)  # Combine top segments for context
    
    return best_answer, max_similarity

# Main interaction loop
print("Ask me anything!")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    answer, accuracy = find_answer_from_raw_data(user_query, segments, segment_embeddings)
    print(f"Bot: {answer}")
    print(f"Accuracy: {accuracy:.2f}")
