import numpy as np
import pandas as pd
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, GRU, Dense, Bidirectional, Conv1D, GlobalMaxPooling1D, 
    BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sentence_transformers import SentenceTransformer, util
import nltk
import os

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load and preprocess data
data_file = '../Data/dataAI.json'
if os.path.exists(data_file):
    data = pd.read_json(data_file)
else:
    data = pd.DataFrame({'question': [], 'answer': []})

questions = data['question']
answers = data['answer']

# Text cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans the input text by removing non-alphanumeric characters, converting to lowercase, 
    and lemmatizing words while excluding stopwords."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    )

# Apply text cleaning
questions = questions.apply(clean_text)

# Encode answers and tokenize questions
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)

# Tokenizer for questions
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences to uniform length
max_sequence_length = max(len(seq) for seq in question_sequences) if question_sequences else 10
padded_sequences = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')

# Train-test split
if len(padded_sequences) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, encoded_answers, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = [], [], [], []

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_sequence_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Bidirectional(GRU(512, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(len(np.unique(encoded_answers)), activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
]

# Train the model if data exists
if len(X_train) > 0:
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

# Load SentenceTransformer model for similarity matching
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = sentence_model.encode(data['question'], convert_to_tensor=True)

def update_dataset(new_question, new_answer):
    """Adds new question-answer pairs to the dataset."""
    global data, question_embeddings, model
    data.loc[len(data)] = {'question': new_question, 'answer': new_answer}
    data.to_json(data_file, orient='records', lines=False)
    question_embeddings = sentence_model.encode(data['question'], convert_to_tensor=True)

def get_similar_response(query, threshold=0.6):
    """Finds the most similar response based on cosine similarity."""
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, question_embeddings).numpy().flatten()
    closest_idx = cosine_scores.argmax()
    if cosine_scores[closest_idx] > threshold:
        return data['answer'][closest_idx]
    return None

def chatbot_response(query):
    """Generates a chatbot response for the given query."""
    greetings = ["hi", "hello", "hey", "greetings", "what's up"]
    query_clean = clean_text(query)

    # Check for greetings
    if any(greet in query_clean.split() for greet in greetings):
        return "Hello! How can I assist you today?"

    # Check for similar responses
    similar_response = get_similar_response(query)
    if similar_response:
        return similar_response

    # Predict using the trained model
    if len(padded_sequences) > 0:
        query_sequence = tokenizer.texts_to_sequences([query_clean])
        query_padded = pad_sequences(query_sequence, maxlen=max_sequence_length, padding='post')
        prediction = model.predict(query_padded)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        if np.max(prediction) < 0.5:
            return "I'm not sure I understand that. Could you try asking in another way?"
        return predicted_label[0]

    return "I currently don't have enough data to answer that. Can you help me learn?"

# Chatbot loop for interaction
if __name__ == "__main__":
    print("Chatbot is running. Type 'Exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'bye', 'quit']:
            print("Chatbot: Thank you for using! Goodbye!")
            break

        response = chatbot_response(user_input)
        print("Chatbot:", response)

        # Allow user to provide feedback
       
        feedback = input("Was this helpful? (yes/no): ").lower()
        if feedback == 'no':
            correct_answer = input("What should I have said?: ")
            update_dataset(user_input, correct_answer)
            print("Thank you! I've learned something new.")
