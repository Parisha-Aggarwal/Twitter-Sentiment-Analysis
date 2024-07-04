import streamlit as st
import pandas as pd
import re
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
import base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image:  url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('twitter.jpg')

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    # st.success('Model loaded successfully')
except Exception as e:
    st.error(f'Error loading model: {e}')

try:
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    # st.success('Vectorizer loaded successfully')
except Exception as e:
    st.error(f'Error loading vectorizer: {e}')

# Initialize the PorterStemmer
port_stem = PorterStemmer()

def stemming(content):
    try:
        stemmed_content = re.sub('[^a-zA-Z]', " ", content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    except Exception as e:
        st.error(f'Error in stemming function: {e}')
        return ""

# Streamlit app
st.title('Twitter Sentiment Analysis')

st.write('Enter the text you want to analyze:')

user_input = st.text_area('Enter text here', '')

if st.button('Analyze'):
    if user_input:
        stemmed_input = stemming(user_input)
        st.write(f'Stemmed Input: {stemmed_input}')  # Debugging line

        try:
            input_transformed = vectorizer.transform([stemmed_input])
            # st.write(f'Transformed Input Shape: {input_transformed.shape}')  # Debugging line
            predicted_sentiment = model.predict(input_transformed)
            if predicted_sentiment[0] == 0:
                st.write('Negative sentiment')
            else:
                st.write('Positive sentiment')
        except Exception as e:
            st.error(f'Error in prediction: {e}')
    else:
        st.write('Please enter some text to analyze.')
