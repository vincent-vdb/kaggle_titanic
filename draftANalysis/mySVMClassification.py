import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

from sklearn import svm

from sklearn import preprocessing
from sklearn import cross_validation


#function to make up the date (fill in the blanks and make features more easy to use)
def makeupdata(df):
  # all sex info are correctly filled up, change them to 0 and 1
  df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  # some ages are not given, give them the mean age
  mean_age = df['Age'].dropna().mean()
  df.Age[df.Age.isnull()] = mean_age

  # make unknown embarked from Cherbourg
  df.Embarked[df.Embarked.isnull()] = 'C'

  # put a reasonable value to the missing fare
  df.Fare[df.Fare.isnull()] = 8.

  # convert embarked info to int
  df['Embarked'] = df['Embarked'].map({'S':-1, 'Q':0, 'C':1}).astype(int)

  #put siblings to 0s or 1s
#  df['SibSp'][np.where(df['SibSp'] > 1)] = 1

  #put parents to 0s or 1s
#  df['Parch'][np.where(df['SibSp'] > 1)] = 1

  ### play with the names to get the titles
  # map the titles as Mister : 0, Mistress : 1, Miss : 2, Others : 3
  for i in range(len(df.Name)):
    if df.Name[i].split(',')[1].split()[0] == 'Mr.':
      df.Name[i] = 0
    elif df.Name[i].split(',')[1].split()[0] == 'Mrs.':
      df.Name[i] = 1
    elif df.Name[i].split(',')[1].split()[0] == 'Miss.':
      df.Name[i] = 2
    else :
      df.Name[i] = 3


  df['Fsize'] = df['Parch'] + df['SibSp']


  return df

# function to rescale the data
def rescaleData (df_train, df_test):
  
  # all sex info are correctly filled up, change them to 0 and 1
  fullDf = pd.concat([df_train, df_test], axis = 0)

  #rescale the age to the mean with unit variance
  mean_age = np.mean(fullDf['Age'])
  var_age = np.var(fullDf['Age'])

  df_train['Age'] = 1.*(df_train['Age'] - mean_age)/var_age
  df_test['Age'] = 1.*(df_test['Age'] - mean_age)/var_age

  #rescale the fare to the median with unit variance
  median_fare = np.median(fullDf['Fare'])
  var_fare = np.var(fullDf['Fare'])
  df_train['Fare'] = 1.*(df_train['Fare'] - median_fare)/var_fare
  df_test['Fare'] = 1.*(df_test['Fare'] - median_fare)/var_fare


  return df_train, df_test


# function to perform logistic regression, random forest classification and SVM classification
def performSVMClassifications(X, Y):

  # Perform the logistic regression
  svmclass = svm.SVC(kernel='rbf', degree = 2, max_iter=-1, random_state=1)
  svmclass.fit(X,Y)
  #print(logistic.coef_)

  scores = cross_validation.cross_val_score(
      svmclass,
      X,
      Y,
      cv=3
  )

  print("scores for logistic regression: ", scores.mean())

  return svmclass



# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe

# fill missing data, and rescale features when needed
df = makeupdata(df)
df_test = makeupdata(df_test)
df, df_test = rescaleData(df, df_test)

#print(df)


# select the features
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Fsize"]

# Perform the classification
logistic = performSVMClassifications(df[predictors],df["Survived"])

# perform the first round of predictions on train dataset
predictionLogistic = logistic.predict(df[predictors])

# same on test dataset
predictionLogisticTest = logistic.predict(df_test[predictors])


# write down the predictions in a csv file to check out
ids = df_test['PassengerId'].values
generalpredictions_file = open("mySVMPredictions.csv", "wb")
open_file_object = csv.writer(generalpredictions_file)
open_file_object.writerow(["PassengerId","Survived" ])
open_file_object.writerows(zip(ids, predictionLogisticTest.astype(int)))
generalpredictions_file.close()

