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
        
        similarity = cosine_similarity(query_embedding.numpy(), question_embedding.numpy())[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = answer
    
    return best_answer, max_similarity

# Function to complete code
def complete_code(unfinished_code):
    # Expanded templates
    code_templates = {
    "model = Sequential()": "model = Sequential()\nmodel.add(Dense(64, activation='relu'))",
    
    "switch (day)": """switch (day) {\n    case 1: printf("Monday"); break;\n    case 2: printf("Tuesday"); break;\n    default: printf("Invalid day");\n}""",
    
    "for(i=0;i<n;i+": "for (i = 0; i < n; i++) {\n    // Complete your logic here\n}",
    
    "if (x > 0": "if (x > 0) {\n    printf(\"Positive number\");\n} else {\n    printf(\"Non-positive number\");\n}",
    
    "def my_function(arg1, arg2": "def my_function(arg1, arg2):\n    # Function implementation here\n    return result",
    
    "while (x < 10": "while (x < 10) {\n    x++;\n    // Add your logic here\n}",
    
    "def merge_sort(arr": """def merge_sort(arr):\n    if len(arr) > 1:\n        mid = len(arr) // 2\n        L = arr[:mid]\n        R = arr[mid:]\n\n        merge_sort(L)\n        merge_sort(R)\n\n        i = j = k = 0\n\n        while i < len(L) and j < len(R):\n            if L[i] < R[j]:\n                arr[k] = L[i]\n                i += 1\n            else:\n                arr[k] = R[j]\n                j += 1\n            k += 1\n\n        while i < len(L):\n            arr[k] = L[i]\n            i += 1\n            k += 1\n\n        while j < len(R):\n            arr[k] = R[j]\n            j += 1\n            k += 1""",
    
    "matrix multiplication": """def matrix_multiplication(A, B):\n    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]\n\n    for i in range(len(A)):\n        for j in range(len(B[0])):\n            for k in range(len(B)):\n                result[i][j] += A[i][k] * B[k][j]\n\n    return result""",
    
    "fibonacci": """# Recursive Fibonacci\ndef fibonacci_recursive(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)\n\n# Iterative Fibonacci\ndef fibonacci_iterative(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a""",
    
    "quick sort": """def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"""
}

    # Match code templates
    for key in code_templates.keys():
        if unfinished_code.strip().startswith(key):
            return code_templates[key]
    
    # Default fallback
    return "I'm not sure how to complete this code. Could you provide more context or rephrase your input?"

# Main interaction loop
print("Ask me anything from your dataset or request code completion by typing 'complete the code' followed by the unfinished code!")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    if user_query.lower().startswith("complete the code"):
        unfinished_code = user_query[len("complete the code"):].strip()
        completed_code = complete_code(unfinished_code)
        print(f"Bot: {completed_code}")
    else:
        answer, accuracy = find_answer(user_query, dataset)
        print(f"Bot: {answer}")
        print(f"Accuracy: {accuracy:.2f}")

        if accuracy < 0.70:
            print("Bot: My confidence is low for this answer. Could you clarify or rephrase your question?")