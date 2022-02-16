# Project 2 - Disaster Response Pipeline Project

[![N|Solid](https://www.python.org/static/community_logos/python-powered-w-70x28.png)](https://www.python.org/)
[![N|Solid](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)](https://scikit-learn.org/stable/)

## Project Objective

The objetive of this project is to develop a disaster classification web app which given a message will classify the message in several disaster categories. The application also shows some statitics on the data used to train the model such as a genre count (where the message was generated), the top percentil category count and the bottom category count. The top and bottom category count could be used to better allocate personal to each disaster provided we were able to normalize its value by the number of personel per category.

## Project Components

The project can be broken down in three components as follows:
1. ETL Pipeline

A Extract Transform Load (ETL) Pipeline which loads the messages and categories datasets, merge them together, cleans the data and stores it in a SQLite database.

2. ML Pipeline

A Machine Learning (ML) Pipeline which loads the data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline using a MultiOutputClassifier based on a RandomForestClassifier which is run through a GridSearchCV which trains and tunes the model and finally exports the final model as pickle file (WARNING: It does take a long time to run that on my machine)

3. Flask Web App

A flask web app which shows the three visualizations explained in the project objective and runs the ML model on the message input in order to classify the message in one of the 36 possible disaster classifications.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    a. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterMsgDb.db`
        
    b. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMsgDb.db models/rfc.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://192.168.0.14:3001

## About Me

📈 Financial market professional with the following certifications:
* CGA - Anbima Asset Manager Certification (Certificado de Gestor Anbima)
* CNPI - Apimec Certified Financial Analyst (Certificado Nacional do Profissional de Investimento)

💻 Machine Learning & Programming
* Py Let's Code - Python Data Science course with more than 400 hours (Let's Code)
* Udacity - Data Scientist Nanodegree - _Ongoing_
* Datacamp - _Ongoing_

