import streamlit as st
import pickle 
import string
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer

cv=pickle.load(open('models/count_vectorizer.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))
model=pickle.load(open('models/model.pkl','rb'))

ps=PorterStemmer()

## creating a function to transform the text -->
def transform_text(text):
    text=text.lower() ## lower case
    text=nltk.word_tokenize(text)  ## tokenization
    y=[]
    for i in text:  ## removing special characters
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:  ## removing stopwords and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) 
            
    text=y[:]
    y.clear()
    
    for i in text:  ## stemming
        y.append(ps.stem(i))
    
    return " ".join(y)

# title of the page
st.title("Email/SMS Spam Classifier")

input_sms=st.text_area("Enter the message")
    
## Predict button
if st.button("Predict"):

    ## 1. Preprocess
    transformed_sms = transform_text(input_sms)
    transformed_sms=[transformed_sms]

    ## 2. Vectorize
    vectorized_sms=cv.transform(transformed_sms)

    ## 3. Predict
    result = model.predict(vectorized_sms)[0]

    ## 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Non Spam")
        