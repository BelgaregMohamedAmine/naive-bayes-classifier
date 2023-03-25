# Project Title

***Example of a spam classifier using Naïve Bayes Algorithm (Machine Learning), Python and Sickit Learn***

The Naive Bayes Classifier project is a machine learning project that uses the Naive Bayes algorithm to classify emails as either spam or ham (non-spam). The project is designed to demonstrate the use of natural language processing (NLP) techniques to preprocess textual data and then use the resulting data to train a machine learning model.

The project involves reading in emails from two directories, one containing spam emails and the other containing ham emails. The emails are preprocessed to extract only the body of the email, and the resulting text is converted into numerical vectors using the CountVectorizer method from scikit-learn. The resulting vectors are then used to train a Multinomial Naive Bayes classifier, which can be used to predict whether a new email is spam or ham.

The project includes several features, including:

-Loading emails from a directory and preprocessing them
-Splitting the data into training and testing sets
-Converting the text data into numerical vectors using CountVectorizer
-Training a Multinomial Naive Bayes classifier on the training data
-Predicting the classification of the testing data using the trained classifier
-Displaying the predictions and true labels for the testing data in a table
-Calculating the prediction score for the testing data
-Generating a confusion matrix to evaluate the performance of the classifier

The project is implemented in Python using various libraries such as scikit-learn, pandas, and matplotlib. It is a useful example of how to apply NLP techniques to machine learning problems and how to evaluate the performance of a classifier using a confusion matrix. The code for the project is available on GitHub, making it accessible to anyone interested in learning how to implement a Naive Bayes classifier for email classification.


