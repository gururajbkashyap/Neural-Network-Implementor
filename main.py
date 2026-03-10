from flask import Flask, render_template, request, jsonify, session
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
from functools import lru_cache
import os
import re
from database import ChatDatabase
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Initialize database
db = ChatDatabase()

# Load dataset with error handling
try:
    with open('data.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} Q&A pairs from dataset")
except FileNotFoundError:
    logger.error("data.json file not found")
    dataset = []
except json.JSONDecodeError as e:
    logger.error(f"Error parsing data.json: {e}")
    dataset = []

# Load additional code examples
try:
    with open('additional_code_examples.json', 'r', encoding='utf-8') as f:
        additional_examples = json.load(f)
    dataset.extend(additional_examples)
    logger.info(f"Loaded {len(additional_examples)} additional code examples")
except FileNotFoundError:
    logger.warning("additional_code_examples.json not found, skipping additional examples")
except json.JSONDecodeError as e:
    logger.error(f"Error parsing additional_code_examples.json: {e}")

# Global variables for models
bert_tokenizer = None
bert_model = None
question_embeddings = None

@lru_cache(maxsize=1000)
def get_bert_embeddings(text):
    """Get BERT embeddings for text with caching"""
    if bert_tokenizer is None or bert_model is None:
        return np.zeros((1, 768))
    
    try:
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            # Ensure 2D shape for cosine similarity
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        return embeddings
    except Exception as e:
        logger.error(f"Error getting BERT embeddings: {e}")
        return np.zeros((1, 768))

def load_models():
    """Load BERT model and tokenizer with error handling"""
    global bert_tokenizer, bert_model, question_embeddings
    
    try:
        logger.info("Loading BERT model...")
        bert_model_name = "bert-base-uncased"
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_model = BertModel.from_pretrained(bert_model_name)
        bert_model.eval()  # Set to evaluation mode
        
        # Precompute embeddings for all questions
        logger.info("Precomputing question embeddings...")
        question_embeddings = []
        for item in dataset:
            question = item['question']
            embedding = get_bert_embeddings(question)
            question_embeddings.append(embedding)
        
        question_embeddings = np.vstack(question_embeddings)
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def get_fallback_answer(query):
    """Provide fallback answers for questions not in the database"""
    query_lower = query.lower()
    
    # Code/Implementation questions - provide actual code examples
    if any(word in query_lower for word in ['implement', 'code', 'example', 'sample', 'write', 'create', 'build', 'develop', 'programming', 'program', 'script', 'tutorial', 'guide']):
        if any(word in query_lower for word in ['cnn', 'convolutional', 'convolution']):
            return """Here's a complete CNN implementation example:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```"""
        
        elif any(word in query_lower for word in ['rnn', 'lstm', 'recurrent', 'gru']):
            return """Here's a complete LSTM implementation example:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load and preprocess data
max_features = 20000
max_len = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Build LSTM model
model = Sequential([
    tf.keras.layers.Embedding(max_features, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```"""
        
        elif any(word in query_lower for word in ['gan', 'generative', 'adversarial']):
            return """Here's a complete GAN implementation example:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten
import numpy as np

# Generator
def build_generator():
    model = Sequential([
        Dense(7*7*256, use_bias=False, input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        Flatten(),
        Dense(1)
    ])
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Training loop
def train_gan(generator, discriminator, epochs=100):
    for epoch in range(epochs):
        # Train discriminator
        noise = tf.random.normal([32, 100])
        generated_images = generator(noise, training=True)
        # ... (complete training loop)
        print(f'Epoch {epoch + 1}/{epochs}')
```"""
        
        else:
            return """Here's a general neural network implementation template:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Create sample data
X = np.random.random((1000, 20))
y = np.random.randint(0, 2, (1000, 1))

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X[:5])
print(predictions)
```

For specific implementations, try asking about:
- CNN for image classification
- LSTM for time series
- GAN for image generation
- Autoencoder for dimensionality reduction"""
    
    # General neural network questions
    elif any(word in query_lower for word in ['what', 'is', 'neural', 'network']):
        return "A neural network is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) that process information using a connectionist approach to computation. Neural networks are particularly effective for pattern recognition, classification, and prediction tasks."
    
    # Machine learning questions
    elif any(word in query_lower for word in ['machine', 'learning', 'ml']):
        return "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn patterns and make predictions."
    
    # Deep learning questions
    elif any(word in query_lower for word in ['deep', 'learning']):
        return "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition."
    
    # General AI questions
    elif any(word in query_lower for word in ['ai', 'artificial', 'intelligence']):
        return "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses various techniques including machine learning, deep learning, natural language processing, computer vision, and robotics."
    
    # Default response
    else:
        return f"I don't have specific information about '{query}' in my knowledge base. I specialize in neural networks, machine learning, and deep learning topics. You could try asking about:\n- Different types of neural networks (CNN, RNN, LSTM, GAN, etc.)\n- Implementation examples and code\n- Machine learning concepts\n- Deep learning architectures\n\nFeel free to rephrase your question or ask about neural network topics!"

def format_code_response(content):
    """Format code blocks with proper syntax highlighting"""
    # Find code blocks
    code_pattern = r'```(\w+)?\n(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) or 'python'
        code = match.group(2)
        
        # Create code block with copy button
        return f'''
<div class="code-block">
    <div class="code-header">
        <span class="code-language">{language}</span>
        <button class="copy-button" onclick="copyCode(this)">
            <i class="fas fa-copy"></i> Copy
        </button>
    </div>
    <pre><code class="language-{language}">{code}</code></pre>
</div>'''
    
    # Replace code blocks
    formatted_content = re.sub(code_pattern, replace_code_block, content, flags=re.DOTALL)
    
    return formatted_content

def enhance_response(response, confidence):
    """Enhance response (confidence indicators hidden from UI)"""
    return response

def find_answer(query, dataset, threshold=0.4):
    """Find the most relevant answer using precomputed embeddings with improved matching"""
    if question_embeddings is None or len(dataset) == 0:
        return "I'm sorry, I don't have any knowledge base loaded.", 0.0
    
    try:
        query_embedding = get_bert_embeddings(query)
        
        # Calculate similarities with all precomputed embeddings
        similarities = cosine_similarity(query_embedding, question_embeddings)[0]
        
        # Get top 3 most similar questions for better matching
        top_indices = np.argsort(similarities)[::-1][:3]
        top_similarities = similarities[top_indices]
        
        # Check for exact keyword matches first (for better accuracy)
        query_lower = query.lower()
        keyword_matches = []
        
        # Define keyword synonyms for better matching
        keyword_synonyms = {
            'cnn': ['convolutional', 'convolution', 'cnn'],
            'code': ['implement', 'implementation', 'example', 'sample', 'code', 'write', 'create', 'build', 'develop', 'programming', 'program', 'script', 'tutorial', 'guide'],
            'rnn': ['recurrent', 'rnn', 'lstm', 'gru'],
            'gan': ['generative', 'adversarial', 'gan'],
            'neural': ['neural', 'network', 'deep', 'learning'],
            'stock': ['stock', 'price', 'prediction', 'forecasting'],
            'time': ['time', 'series', 'sequence', 'temporal'],
            'python': ['python', 'tensorflow', 'keras', 'pytorch', 'numpy', 'pandas', 'sklearn'],
            'tensorflow': ['tensorflow', 'keras', 'tf'],
            'pytorch': ['pytorch', 'torch']
        }
        
        # Check if query is code-related and prioritize code responses
        is_code_query = any(word in query_lower for word in ['implement', 'code', 'example', 'sample', 'write', 'create', 'build', 'develop', 'programming', 'program', 'script', 'tutorial', 'guide'])
        
        # First, try exact question matching (highest priority)
        for i, item in enumerate(dataset):
            question_lower = item['question'].lower()
            if query_lower == question_lower:
                return item['answer'], 1.0  # Perfect match
        
        # Then try partial question matching
        for i, item in enumerate(dataset):
            question_lower = item['question'].lower()
            if query_lower in question_lower or question_lower in query_lower:
                return item['answer'], 0.95  # Very high confidence for partial matches
        
        # Then check for keyword matches
        for i, item in enumerate(dataset):
            question_lower = item['question'].lower()
            answer_lower = item['answer'].lower()
            
            # Check for direct keyword matches
            query_words = query_lower.split()
            match_score = 0
            
            for word in query_words:
                if len(word) > 2:
                    # Direct match in question (higher weight)
                    if word in question_lower:
                        match_score += 0.4
                    if word in answer_lower:
                        match_score += 0.2
                    
                    # Special boost for code-related queries
                    if is_code_query and any(code_word in word for code_word in ['code', 'implement', 'example', 'sample']):
                        if word in question_lower:
                            match_score += 0.3  # Extra boost for code queries
                        if word in answer_lower:
                            match_score += 0.2
                    
                    # Synonym match
                    for category, synonyms in keyword_synonyms.items():
                        if word in synonyms:
                            for synonym in synonyms:
                                if synonym in question_lower:
                                    match_score += 0.5
                                if synonym in answer_lower:
                                    match_score += 0.3
            
            if match_score > 0.6:  # Higher threshold for keyword matches
                keyword_matches.append((i, min(match_score, 0.95)))
        
        # If we have keyword matches, use the best one
        if keyword_matches:
            best_match = max(keyword_matches, key=lambda x: x[1])
            return dataset[best_match[0]]['answer'], best_match[1]
        
        # Use semantic similarity as fallback
        best_similarity = max(top_similarities)
        best_index = top_indices[np.argmax(top_similarities)]
        
        if best_similarity > threshold:
            return dataset[best_index]['answer'], float(best_similarity)
        else:
            # If no good match found, use fallback
            fallback_answer = get_fallback_answer(query)
            return fallback_answer, 0.3  # Low confidence for fallback
    
    except Exception as e:
        logger.error(f"Error in find_answer: {e}")
        return "I'm sorry, I encountered an error while processing your question.", 0.0

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Neural Network Chatbot is running'})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        session_id = data.get('session_id')
        user_id = session.get('user_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Find answer using improved matching
        answer, similarity = find_answer(query, dataset)
        
        # Format code blocks - NEWLY ADDED
        formatted_answer = format_code_response(answer)
        
        # Enhance response
        enhanced_answer = enhance_response(formatted_answer, similarity)
        
        # Save messages to database if user is logged in
        if session_id and user_id:
            db.add_message(session_id, 'user', query)
            db.add_message(session_id, 'assistant', enhanced_answer, float(similarity))
        
        response = {
            'answer': enhanced_answer,
            'accuracy': float(similarity),
            'query': query,
            'session_id': session_id
        }
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# User authentication endpoints
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        existing_user = db.get_user(username)
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 400
        
        # Create user
        user_id = db.create_user(username, password)
        session['user_id'] = user_id
        session['username'] = username
        
        return jsonify({'message': 'User created successfully', 'user_id': user_id})
    
    except Exception as e:
        logger.error(f"Error in register endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Get user
        user = db.get_user(username)
        if not user or user[2] != password:  # user[2] is password
            return jsonify({'error': 'Invalid username or password'}), 401
        
        session['user_id'] = user[0]  # user[0] is user_id
        session['username'] = username
        
        return jsonify({'message': 'Login successful', 'user_id': user[0]})
    
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logout successful'})

# Session management endpoints
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    sessions = db.get_user_sessions(user_id)
    return jsonify(sessions)

@app.route('/api/sessions', methods=['POST'])
def create_session():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session_id = db.create_session(user_id, 'New Chat')
    return jsonify({'session_id': session_id, 'title': 'New Chat'})

@app.route('/api/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    messages = db.get_session_messages(session_id)
    return jsonify(messages)

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    success = db.delete_session(session_id)
    if success:
        return jsonify({'message': 'Session deleted successfully'})
    else:
        return jsonify({'error': 'Session not found or access denied'}), 404

@app.route('/api/sessions/<session_id>/title', methods=['PUT'])
def update_session_title(session_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    title = data.get('title', '').strip()
    
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    
    success = db.update_session_title(session_id, title)
    if success:
        return jsonify({'message': 'Title updated successfully'})
    else:
        return jsonify({'error': 'Session not found or access denied'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
