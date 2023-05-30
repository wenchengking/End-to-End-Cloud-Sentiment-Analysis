import os
import pickle
from pathlib import Path

import aws_utils as aws
import boto3
import numpy as np
import streamlit as st

# Load the Iris dataset and trained classifier
BUCKET_NAME = os.getenv("BUCKET_NAME", "msia423-group8-artifact")
#ARTIFACTS_PREFIX = Path(os.getenv("ARTIFACTS_PREFIX", "artifacts/"))

# Create artifacts directory to keep model files
#artifacts = Path() / "artifacts"
#artifacts.mkdir(exist_ok=True)

@st.cache_data #cache data
def load_model_versions(bucket_name: str):
    """model version is listed as prefix in s3 bucket"""
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    folders = []
    for prefix in response.get('CommonPrefixes', []):
        folder = prefix['Prefix'].rstrip('/')
        folders.append(folder)

    return folders


@st.cache_data #cache data
def load_data(data_file, s3_key):
    #print("Loading artifacts from: ", artifacts.absolute())
    # Download files from S3
    aws.download_s3(BUCKET_NAME, s3_key, data_file)

    with open(data_file, 'rb') as file:
        vectorizer = pickle.load(file)

    return vectorizer

@st.cache_resource #cache resource
def load_model(model_file, s3_key):
    # Download files from S3
    aws.download_s3(BUCKET_NAME, s3_key, model_file)

    with open(model_file, 'rb') as file:
        clf = pickle.load(file)

    return clf

# Create the application title and description
st.title("Hotel Review Sentiment Analysis 🏨")

with st.expander("Team Members - Group 8 👥"):
    st.write("""
    - **Brian Hong** - [Introduction Page](https://www.mccormick.northwestern.edu/analytics/people/students/class-of-2023/hong-brian.html)
    - **Yiyue (Jessie) Xu** - [Introduction Page](https://www.mccormick.northwestern.edu/analytics/people/students/class-of-2023/xu-yiyue.html)
    - **Wencheng Zhang** - [Introduction Page](https://www.mccormick.northwestern.edu/analytics/people/students/class-of-2023/zhang-wencheng.html)
    - **Zhengyuan (Donald) Li** - [Introduction Page](https://www.mccormick.northwestern.edu/analytics/people/students/class-of-2023/li-zhengyuan.html)
    """)


with st.expander("Learn more about the app 📖"):
    st.write("""
    This app was trained using a Logistic Regression model, one of the simplest yet effective machine learning models for classification tasks. 
    The model was trained on a large dataset of hotel reviews, which were preprocessed and transformed into numerical features using TF-IDF Vectorizer. 
    This allows the model to understand the semantic meaning of the words in the review, and predict whether the review is positive (satisfied) or negative (unsatisfied). 
    Please note that the model might not always be 100% accurate, and should be used as a guide rather than a definitive judgment.
    
     - **Project Code** - [GitHub](https://github.com/wenchengking/Cloud_Engineering_DS)

    """)


st.subheader("Model Selection 🔄")
model_version = os.getenv("DEFAULT_MODEL_VERSION", "default")

# Find available model versions in artifacts dir
available_models = load_model_versions(BUCKET_NAME)

# Create a dropdown to select the model
model_version = st.selectbox("Select Model", list(available_models))
st.write(f"Selected model version: {model_version}")


# Establish the dataset and TMO locations based on selection
version_model_dir = Path() / model_version
#version_model_dir = model_version
version_model_dir.mkdir(exist_ok=True)

review_vec_file = version_model_dir / "vectorizer.pkl"
review_model_file = version_model_dir / "inference_model.pkl"

# Configure S3 location for each artifact
#review_s3_key = str(model_version / review_vec_file.name)
#review_model_s3_key = str(model_version / review_model_file.name)

review_s3_key = os.path.join(model_version, review_vec_file.name)
review_model_s3_key = os.path.join(model_version, review_model_file.name)

# Load the dataset and TMO into memory
vectorizer = load_data(review_vec_file, review_s3_key)
clf = load_model(review_model_file, review_model_s3_key)


# create a text box for user input. My model will predict the comments. Add a bottom to triger the prediction
user_input = st.text_input("Enter your value: 🔤")
submit = st.button('Submit 🚀 🚀 🚀')

# Make predictions on user inputs
input_data = vectorizer.transform([user_input])
prediction = clf.predict(input_data)

#change prediction = 0 to unsasticfied and 1 to satisfied
if prediction == 0:
    prediction_bin = "unsatisfied"
else:
    prediction_bin = "satisfied"

# Display the predicted class and probability
st.subheader("Prediction 🔮")
if prediction_bin == "unsatisfied":
    st.markdown(f"Our analysis suggests that the sentiment of the customer review is: <span style='color:#FF0000; font-size:30px'>{prediction_bin}</span>", unsafe_allow_html=True)
else:
    st.markdown(f"Our analysis suggests that the sentiment of the customer review is: <span style='color:#00FF00; font-size:30px'>{prediction_bin}</span>", unsafe_allow_html=True)


if prediction_bin == "unsatisfied":
    with st.expander("Suggestions for improvement 💡"):
        st.write("""
        Based on the sentiment of the customer review, here are some suggestions for your business:
        - **Address the customer's concerns:** Directly respond to any negative feedback. Show your customers that you are open to criticism and willing to improve.
        - **Investigate the issue:** Check if the issues mentioned in the review are valid concerns that need to be addressed within your business operations.
        - **Implement changes where necessary:** If the review points towards a certain issue, consider making changes in that area to enhance the customer experience.
        """)

    with st.expander("How to respond to unsatisfied reviews: 📝"):
        st.write("""
        Here are some steps on how to effectively respond to negative reviews:
        - **1. Stay calm and professional:** When responding to a negative review, maintain a professional tone. It's crucial not to appear defensive or dismissive of the customer's concerns.
        - **2. Acknowledge the issue:** Show your customers that you value their feedback by acknowledging their concerns. This first step is crucial in showing that you take customer service seriously.
        - **3. Apologize and empathize:** Provide a sincere apology for any inconvenience the customer has experienced. Demonstrate empathy and understanding for their dissatisfaction.
        - **4. Offer a solution:** After acknowledging the issue, offer a concrete solution. This could be a refund, a replacement, or any other appropriate response to address the issue.
        - **5. Take the conversation offline:** If the issue needs a deeper discussion, suggest moving the conversation to a private channel. This can provide a space for a more detailed conversation, without airing all the specifics publicly.
        - **6. Follow up:** Once the issue has been resolved, follow up with the customer to ensure they're satisfied with the solution provided.
        - **7. Learn from the feedback:** Use negative reviews as an opportunity for improvement. Assess the feedback objectively and use it to make improvements in your services.
        """)
