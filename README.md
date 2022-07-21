# Disaster_Response_Pipeline Project
Repository for Disaster response pipeline project with ETL,NLP and ML pipelines

## Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
 ### Project Motivation
As a part of Datascience nanodegree learning requirement, I have applied my DE skills that I have learnt thru the Udacity platform to perform an analysis on disaster data from appen.com and build a classifier model for classifying messages. This project utilizes an ETL pipeline, a ML pipeline and a Flask web app. 
The ML model categorizes real messages sent during a disaster event and then sent to the corresponding disaster relief agency. 
The web app allows emergency workers to input messages and find out the category of the message. 


### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # ETL/data cleaning pipeline    
|- DisasterResponse.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model in a pickle file     


README.md    

### Components
There are 3 components used for this project. 

#### A. ETL Pipeline
A Python script, `process_data.py`, creates an ETL pipeline which:

 - Loads messages and categories datasets
 - Merges the two datasets
 - Cleans and transforms the data
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation_myversion` was used to do EDA and to prepare the process_data.py python file. 
 
#### B. ML Pipeline
A Python script, `train_classifier.py`, creates a ML pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used in preparation of the train_classifier.py file

#### C. Flask Web App
A web app that a emergency worker can use to input new message and get the output in several categories. The app also contains visualizations for easier understanding of underlying distributions of data

![Screenshot Question Class](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/Screenshot%20Question%20Class.png)

![Screenshot Question Class2](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/Screenshot%20Question%20Class2.png)

![Screenshot Overview Dataset](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/Screenshot%20Overview%20Dataset.png)

![newplot refugees.png](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/newplot%20refugees.png)

![newplot missing ppl.png](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/newplot%20missing%20ppl.png)

![newplot death.png](https://github.com/ashwinibhatt85/Disaster_Response_Pipeline/blob/main/Images/newplot%20death.png)





### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for the bolierplate code and Rajat for his review on the ETL Pipeline prep. 
