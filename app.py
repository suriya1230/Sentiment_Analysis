import pickle
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk


@st.cache_resource
def load_stopword():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model():
    with open('C:\Users\ksrag\OneDrive\Desktop\python work\project birthday\Sentiment Analysis\sentiment_model.pkl','rb')as model_file:
        model=pickle.load(model_file)
    with open('C:\Users\ksrag\OneDrive\Desktop\python work\project birthday\Sentiment Analysis\vectorizer.pkl', 'rb')as vectorizer_file:
        vectorizer=pickle.load(vectorizer_file)
    return model,vectorizer

def predict_sentiment(text,model,vectorizer,stop_words):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ',
    re.sub(r'\s+', ' ', 
    re.sub(r'\d+', '', 
    re.sub(r'[^\w\s]', '', 
    re.sub(r'#\w+', '', 
    re.sub(r'http\S+|www.\S+', '', 
    re.sub(r'@\w+', '',text ))))))).strip()
    
    # Convert to lowercase and split into words
    text = text.lower().split()
    
    # Remove stopwords and apply stemming
    text = [word for word in text if word not in stop_words]
    
    # Join back to single string
    text = ' '.join(text)
    text=[text]
    text=vectorizer.transform(text)

    sentement=model.predict(text)
    return "nagative" if sentement==0 else "positive"


def create_card(sentement):
        color="green" if sentement == "positive" else "red"
        card_html=f""" 
        <div style="background-color:{color};
        padding:10px;
        border-redius:5px;
        margin:10px 0;">
        <h5 style="color: white">{sentement}sentiment</h5>
        """
        return card_html
    
def main():
    st.title("Sentiment Analysis")
    stop_words = load_stopword()
    model, vectorizer = load_model()
   
    option = st.selectbox("Choose an option", ["Input Text"])

    if option == "Input Text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")
            card_html = create_card(sentiment)
            st.markdown(card_html, unsafe_allow_html=True)
        
            
                
if __name__=="__main__":
    main()