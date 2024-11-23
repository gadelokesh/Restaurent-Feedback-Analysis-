# frontend
import streamlit as st 
import pickle

import re
import nltk
from nltk.corpus import stopwords # for stopwords
from nltk.stem.porter import PorterStemmer # for stem the words


model=pickle.load(open(r"C:\Users\gadel\VS Code projects\Restaurant_review_analysis\model.pkl","rb"))
tfidf=pickle.load(open(r"C:\Users\gadel\VS Code projects\Restaurant_review_analysis\tfidf.pkl",'rb'))

st.title("FeedBack Analysis App")

# Set background (optional)
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Ffood-background&psig=AOvVaw13pu1q61drvT6YayHP7LGm&ust=1732439104386000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJiituCM8okDFQAAAAAdAAAAABAE")

st.write("""
### About the App
This *Feedback Analysis App* uses Natural Language Processing (NLP) to analyze customer reviews and predict their sentiment. Just enter any feedback or review, and the app will determine whether the sentiment is positive or negative. This can be helpful for businesses to quickly suggestion customer satisfaction and improve service quality.
""")

inputText=st.text_area("Enter your review Here")

corpus=[]


# take to proper format
for i in inputText:
    review = re.sub('[^a-zA-Z]', ' ', inputText)
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()   
    review = ' '.join(review)
    corpus.append(review)
    
# Transform the text using the tfidf
    review_vector = tfidf.transform([review])  # Transform to the tfidf format

if st.button("Click Here"):
    predict=model.predict(review_vector)
    if predict==1:
        st.success(f"Your feed back is a positive review. ")
    else:
        st.success("Your feed back is a negative review.")