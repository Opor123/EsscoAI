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
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
data = pd.read_json('Data/dataAI.json')

# Extract questions and answers
questions = data['question']
answers = data['answer']

# Clean text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

questions = questions.apply(clean_text)

# Encode answers
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)

# Tokenize questions
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences
max_sequence_length = max(len(seq) for seq in question_sequences)
padded_sequences = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_answers, test_size=0.2)


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

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

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



# Function to test the chatbot
def chatbot_response(query):
    # Try similarity-based matching first
    similar_response = get_similar_response(query)
    if similar_response:
        return similar_response
    # Fallback to model prediction
    query_clean = clean_text(query)
    query_sequence = tokenizer.texts_to_sequences([query_clean])
    query_padded = pad_sequences(query_sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(query_padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Test the chatbot
test_query = input("User: ")
print("Chatbot Response:", chatbot_response(test_query))
