# SMS Spam Classifier

## Overview
Welcome to the SMS Spam Classifier project! This project aims to build a machine learning model capable of classifying SMS messages as either spam or not spam (ham). We've utilized Jupyter Notebook along with Python libraries such as pandas, scikit-learn, numpy, and matplotlib to develop this classifier.

## Dataset
Our dataset comprises SMS messages labeled as either spam or ham. You can find the dataset in the `data` directory named `sms_spam.csv`. It's sourced from [insert dataset source here].

## Usage
To get started with this project, follow these steps:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

2. **Install Dependencies**: Install the necessary Python libraries.

3. **Explore the Dataset**: Dive into the dataset located in the `data` directory. Familiarize yourself with the structure and content of the SMS messages.

4. **Open the Jupyter Notebook**: Launch the Jupyter Notebook `SMS_Spam_Classifier.ipynb` to explore the project code. Execute the code cells step by step to understand the data preprocessing, model training, and evaluation process.

5. **Model Training and Evaluation**: Follow the instructions in the notebook to preprocess the data, train the machine learning model, and evaluate its performance using accuracy score and classification report metrics.

## Libraries Used
We've utilized several Python libraries for this project:
- pandas: For efficient data manipulation and analysis.
- scikit-learn: To implement machine learning algorithms and evaluation metrics.
- numpy: For numerical computing.
- matplotlib: For data visualization.

## Model
Our classifier model is built using the Naive Bayes algorithm. It utilizes a CountVectorizer to convert text data into numerical features and then trains a Multinomial Naive Bayes classifier.

## Results
We evaluate the performance of our model using accuracy score and classification report metrics. For detailed insights into the model's performance, refer to the Jupyter Notebook.
