import streamlit as st
import pickle
import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess(text):
    # Tokenize
    tk = TweetTokenizer()
    text = tk.tokenize(text)
    text = " ".join(text)

    # Remove special characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    # Remove short words
    tokens = word_tokenize(text)
    text = ' '.join([w for w in tokens if len(w) > 2])

    # Stemming
    stemmer = SnowballStemmer('english')
    tokens = word_tokenize(text)
    text = [stemmer.stem(i.lower()) for i in tokens]
    text = ' '.join(text)

    # Remove stopwords
    stop = stopwords.words('english')
    tokens = word_tokenize(text)
    text = [i for i in tokens if i not in stop]
    return ' '.join(text)

# Reverse map
reverse_map = {1: "Positive", -1: "Negative", 0: "Neutral"}

# Streamlit UI
st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet to analyze its sentiment.")

user_input = st.text_area("Tweet Text", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter some text.")
    else:
        # Preprocess input
        processed = preprocess(user_input)

        # Vectorize
        input_vec = vec.transform([processed])

        # Predict
        prediction = model.predict(input_vec)

        # Display result
        sentiment = reverse_map[prediction[0]]
        st.success(f"The sentiment of the tweet is: **{sentiment}**")

        # Optional: show processed text
        st.write("Processed text:", processed)

st.write("\n\n*Model trained on Twitter validation dataset using SVM.*")

if __name__ == "__main__":
    import os
    os.system('python -m streamlit run "' + __file__ + '" --server.headless true')
