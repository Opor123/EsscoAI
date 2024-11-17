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
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sentence_transformers import SentenceTransformer, util
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
data = pd.read_json('dataAI.json')

# Extract questions and answers from the dataset
questions = data['question']
answers = data['answer']

# Clean text function to preprocess the questions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Lemmatize words and remove stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply text cleaning to all questions
questions = questions.apply(clean_text)

# Encode the answers into numerical format
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)

# Tokenize the cleaned questions
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences to ensure uniform input size
max_sequence_length = max(len(seq) for seq in question_sequences)
padded_sequences = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_answers, test_size=0.2)

# Build the sequential model
model = Sequential([
    # Embedding layer to convert word indices to dense vectors
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_sequence_length),
    # Convolutional layer to extract features from sequences
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    # Bidirectional GRU layer for sequence processing
    Bidirectional(GRU(512, return_sequences=True)),
    GlobalMaxPooling1D(),
    # Dense layer with L2 regularization to prevent overfitting
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.4),
    # Output layer with softmax activation for multi-class classification
    Dense(len(np.unique(encoded_answers)), activation='softmax')
])

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train the model with validation
history = model.fit(
    X_train, y_train,
    epochs=256,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Initialize sentence transformer model for similarity matching
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = sentence_model.encode(data['question'], convert_to_tensor=True)

# Function to get the most similar response based on cosine similarity
def get_similar_response(query):
    # Preprocess and encode the query
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    # Calculate similarity scores
    cosine_scores = util.pytorch_cos_sim(query_embedding, question_embeddings).numpy().flatten()
    # Get the index of the closest question
    closest_idx = cosine_scores.argmax()
    # Set a threshold to ensure a high similarity match
    if cosine_scores[closest_idx] > 0.6:
        return data['answer'][closest_idx]
    return None

# Function to test the chatbot responses ```python
def chatbot_response(query):
    # Detect if the input is a greeting
    greetings = ["hi", "hello", "hey", "greetings", "what's up"]
    query_clean = clean_text(query)
    
    # Check if the user input matches any greeting
    if any(greet in query_clean.split() for greet in greetings):
        return "Hello! How can I assist you today?"

    # Try similarity-based matching for responses
    similar_response = get_similar_response(query)
    if similar_response:
        return similar_response

    # Fallback to model prediction if no similar response is found
    query_sequence = tokenizer.texts_to_sequences([query_clean])
    query_padded = pad_sequences(query_sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(query_padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    # Provide a default response if confidence is low
    if np.max(prediction) < 0.5:  # Adjust threshold as needed
        return "I'm not sure I understand that. Could you try asking in another way?"
    
    return predicted_label[0]

# Test the chatbot in a loop for continuous interaction
while True:
    test_query = input("User: ")
    
    # Exit condition for the chatbot
    if test_query in ['Exit', 'Bye', 'q', 'Thank you']:
        print("Chatbot: Thank you for using!!!")
        break

    # Output the chatbot's response to the user's query
    print("Chatbot Response:", chatbot_response(test_query))