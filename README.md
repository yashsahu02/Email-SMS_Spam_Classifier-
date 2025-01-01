# Email/SMS Spam Classifier
A machine learning web application that classifies emails as either spam or not spam based on their content using advanced text processing and classification techniques.

Try it out: <a style="text-decoration: none;" target="_blank" href='https://email-spam-classifierr.streamlit.app/'>Link</a>

<hr>

## Table of Contents
- Introduction
- Features
- Dataset Description
- Model Overview
- Technologies Used
- Installation
- Usage
- Results
- Future Work
- Demo

<hr>

## 1. Introduction
The Email Spam Classifier is a machine learning-based application designed to filter emails into two categories: spam and not spam. By leveraging Natural Language Processing (NLP) and supervised learning algorithms, the system enhances email filtering capabilities and reduces unwanted content.

<hr>

## 2. Features
- Classifies emails and SMS as spam or not spam with high accuracy.
- User-friendly web interface powered by Streamlit.
- Analyzes text features to identify spam patterns effectively.

<hr>

## 3. Dataset Description
The dataset contains email text and labels indicating whether the email is spam (1) or not spam (0).

### Key Features:
- Email Text: The raw content of the email.
- Label: Binary indicator (1 for spam, 0 for not spam).

The dataset used in this project was obtained from Kaggle's Email Spam Dataset.

**Source**: <a href='https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset'>SMS Spam Collection Dataset</a>

<hr>

## 4. Model Overview
Algorithm Used: RandomForestClassifier
#### Preprocessing Steps:
- Text cleaning (removal of special characters, stop words, etc.).
- Feature extraction using TF-IDF or Bag of Words.
#### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score

<hr>

## 5. Technologies Used
#### Programming Language: 
- Python
#### Frameworks and Libraries:
- Streamlit (for the web interface)
- Scikit-learn (for machine learning models)
- NumPy and Pandas (for data manipulation)
- Matplotlib and Seaborn (for visualizations)

<hr>

## 6. Installation
#### Follow these steps to set up the project locally:

1. Clone the repository:<br>
**command:** <code>git clone https://github.com/yashsahu02/Email-SMS_Spam_Classifier-.git</code>

2. Navigate to the project directory:<br>

3. Install the dependencies:<br>
**command:** <code>pip install -r requirements.txt</code>

<hr>

## 7. Usage
#### Run the application:
- **command:** <code>streamlit run app.py</code><br>
Here app.py is name of python file.
#### Use the web interface to:

<!---
- Upload email text files.
-->
- Manually input email text.
- View classification results (spam or not spam).

<hr>

## 8. Results
#### Model Performance:

- Accuracy Score: 0.9719
- Precision Score: 0.9909
- Confusion Matrix:<br> 
 [[895   1]<br>
 [ 28 110]]

<hr>

## 9. Future Work
1. Integrate advanced machine learning models like XGBoost or CatBoost to improve classification performance.
2. Add adaptive learning to allow the model to improve over time based on user feedback.
3. Explore deep learning models like LSTMs or BERT for better text understanding.

<hr>

## 10. Demo
<!--
### Live Demo : <a href='https://www.freeconvert.com/video-compressor/download'>Try It Out</a> 
-->

### Demo Video:
<br>

https://github.com/user-attachments/assets/f1dcfa4f-a2c6-40db-ae4d-78c7be8250a3

<br>

### Screenshot: 
<br>

![Screenshot (17)](https://github.com/user-attachments/assets/61b6de1c-b581-4069-989b-b9b1a2c41d17)

<br>
<br>
<br>

![Screenshot (18)](https://github.com/user-attachments/assets/d7028574-3d59-4648-ae7e-5deff0aa45fb)

<br>
<br>
<br>

![Screenshot (19)](https://github.com/user-attachments/assets/6a681e27-55b0-4cfa-91eb-491983f9a2c1)

<br>
<br>
<br>

![Screenshot (23)](https://github.com/user-attachments/assets/5ca04945-4d13-4bbd-8117-5e5afc2b6b6c)

<br>
<br>
<br>

![Screenshot (24)](https://github.com/user-attachments/assets/7d2c9d3a-2cee-4d81-88fc-e8a8aacf819f)
