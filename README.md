Introduction

This project is focused on building an Email/SMS Spam Classifier. The goal is to classify incoming messages as Spam or Not Spam using Natural Language Processing (NLP) techniques and machine learning models. By leveraging the power of the Multinomial Naive Bayes algorithm, this project aims to offer a reliable prediction model for detecting spam messages. A web-based application is created using Streamlit to allow users to interact with the classifier.
________________________________________
Project Overview

The primary goal of this project is to build a spam detection system that can accurately classify SMS and email messages as spam or not spam. The objectives of this project include:

•	Text preprocessing to clean and prepare data for modeling.

•	Experimenting with different classification models and identifying the best one for the task.

•	Saving the trained model and vectorizer using Pickle for future reuse.

•	Developing a Streamlit app for the user to input messages and receive predictions.

•	Deploying the app on Render for easy accessibility.
________________________________________
Technologies Used

•	Python 3.11: Programming language used for building the project.

•	Streamlit: Web framework for creating the interactive web app.

•	NLTK: Natural Language Toolkit used for text preprocessing.

•	Scikit-learn: Machine learning library used for model building.

•	Pandas: Data manipulation and analysis library.

•	Render: Deployment platform for hosting the web app.

•	Pickle: For saving the trained model and vectorizer.
________________________________________
Understanding the Dataset

The dataset typically includes:

•	Message: The raw text data, which could be an SMS or email message.

•	Label: The target variable, where:

o	1 represents Spam.

o	0 represents Not Spam.
________________________________________
Steps in Jupyter Notebook

Before deploying the model, the dataset was processed and explored in a Jupyter notebook. Here’s a detailed step-by-step of what was done:

1. Importing Libraries

Several libraries were used throughout the project:

•	Pandas for data manipulation.

•	NLTK for text processing.

•	Scikit-learn for building machine learning models.

•	Matplotlib.pyplot and Seaborn for visualizations.

3. Loading Data

The dataset, typically a .csv file, was loaded into a pandas DataFrame for easy manipulation and analysis.

5. Data Cleaning

Data cleaning involved:

•	Dropping unnecessary columns.

•	Handling any missing data to ensure the dataset was clean for processing.

•	Removing duplicates.

•	Label encoding for target values to transform it into numerical labels.


7. Exploratory Data Analysis

Basic statistics and visualizations were generated to understand:

•	The distribution of spam and non-spam messages.

•	The average length of messages in each category.

•	Detecting multicollinearity 

5. Text Data Preprocessing

Preprocessing steps were critical to transforming the raw text into a format that could be fed into the machine learning models. Below are the techniques used:


•Converting Text to Lowercase

The entire dataset was converted to lowercase to avoid issues with case sensitivity.

•Tokenization

Messages were split into individual words (tokens) to facilitate the removal of unnecessary words.

•Removing Special Characters

Special characters and numbers were removed from the tokenized messages, ensuring only relevant text data remained.

•Removing Stop Words and Punctuation

Stop words (common but irrelevant words such as “the”, “and”) and punctuation were removed to reduce noise in the data.

•Stemming

Stemming was applied to reduce words to their root form. For example, “running” would be reduced to “run”, thereby standardizing similar words.

6. Model Building

After preprocessing the data, multiple classification algorithms were tested:

Models Used

•	Logistic Regression

•	Support Vector Machine (SVM)

•	Random Forest

•	Naïve Bayes models

•	K Neighbors

•	AdaBoost

•	Bagging Classifier

•	Extra Trees Classifier

•	GradientBoosting

•	XGB Classifier

The best-performing model, based on accuracy and precision , was found to be Multinomial Naive Bayes.

7. Saving the Model and Vectorizer

The trained model and vectorizer were saved using the Pickle module. This step allowed the model and vectorizer to be loaded in the web app without the need to retrain them.

•	The model was saved as model.pkl.

•	The vectorizer was saved as vectorizer.pkl.
________________________________________
Building the Streamlit App in PyCharm

Once the model and vectorizer were ready, the focus shifted to creating a Streamlit web application for real-time predictions. Here’s a breakdown of the steps followed:

1. Setting Up PyCharm

•	A new project folder was created in PyCharm.

•	The streamlit package was installed for creating the web app.

2. Writing the Streamlit Code

The Streamlit app was designed to:

•	Take in a message from the user.

•	Preprocess the message using the same logic as in the Jupyter notebook.

•	Use the trained vectorizer to convert the input message into a format the model can understand.

•	Use the saved model to predict whether the message is spam or not.

3. Testing the Streamlit App Locally

Before deployment, the app was tested locally by running streamlit run app.py. This allowed for adjustments and improvements to the user interface and backend logic.
________________________________________
Pushing the Code to GitHub

Once the Streamlit app was ready, the project was pushed to a GitHub repository. The following steps were used:

1.	Initialize Git Repository: Initialize the folder as a Git repository.

2.	Commit the Changes: Stage and commit all files.

3.	Push to GitHub: Push the project to a newly created GitHub repository.
________________________________________
Deploying the Project Using Render

Render was chosen as the deployment platform for its ease of use. The steps followed for deployment were:

1.	Connecting to GitHub: Render was linked to the GitHub repository containing the project.

2.	Creating a Web Service: A new Streamlit service was created on Render.

3.	Deploying the App: After specifying the branch and deployment settings, the app was deployed live.
________________________________________
Challenges Faced

One significant challenge encountered during deployment was related to missing NLTK resources, particularly the stopwords dataset. This resulted in the following error:

LookupError: Resource 'stopwords' not found. Please use the NLTK Downloader to obtain the resource: >>> import nltk >>> nltk.download('stopwords')
This issue was resolved by ensuring that the necessary NLTK datasets were downloaded and available during both local testing and deployment.
________________________________________
Conclusion

This project successfully demonstrates the application of machine learning and natural language processing techniques to detect spam in SMS and email messages. The web application provides an easy-to-use interface for users to classify messages, showcasing the capabilities of Streamlit for rapid application development. Future enhancements could include improving model accuracy, expanding the dataset, and adding features for user feedback on predictions.
