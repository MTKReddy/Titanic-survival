# Titanic-survival
Predicting the surival of passenger using logistic regression in python

### Files included :
	* Titanic_train.csv - Training dataset
	* titanic_test.csv - Testing dataset
	* my titanic disaster.py - python script that extracts the predictions as a CSV file
	* my titanic disaster traning model - python script for training the model with evaluation metrics
	* my titanic disaster.ipynb - python script in jupyter notebook with the required graphical interpretations
	* Titanic_result.csv - output extracted from the python script
	
### Libraries used :
	* Pandas
	* numpy
	* matplotlib
	* seaborn
	* sklearn

#### Problem type : Binary Classification

#### Machine learning algorithm used : Logistic Regression

### Algorithm :
	* importing the required libraries
	* Read and input the training and test data from the CSV files
	* Exploring the data for null values 
	* Exploratory analysis to check for the influence of the various features over the output
	* Cleaning the training and test data
	* processing the categorical values and other unwanted features
	* Separating and assigning the features and output parameters for the training and test dataset
	* Fitting the logistic model using the training data
	* calculate the accuracy using evaluation metrics
	* Use the trained logistic model to predict the survival of passengers in the test data
	* Output the predicted data into a CSV file