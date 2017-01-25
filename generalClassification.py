import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn import preprocessing
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

#from sklearn.neural_network import MLPClassifier


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

  return df

def rescaleData (df_train, df_test):
  
  # all sex info are correctly filled up, change them to 0 and 1
  fullDf = pd.concat([df_train, df_test], axis = 0)

  #rescale the age to the mean with unit variance
  mean_age = np.mean(fullDf['Age'])
  var_age = np.var(fullDf['Age'])

  df_train['Age'] = 1.*(df_train['Age'] - mean_age)/var_age
  df_test['Age'] = 1.*(df_test['Age'] - mean_age)/var_age

  #rescale the fare to the mean with unit variance
  median_fare = np.median(fullDf['Fare'])
  var_fare = np.var(fullDf['Fare'])
  df_train['Fare'] = 1.*(df_train['Fare'] - median_fare)/var_fare
  df_test['Fare'] = 1.*(df_test['Fare'] - median_fare)/var_fare


  return df_train, df_test



# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe

# fill missing data, and rescale features when needed
df = makeupdata(df)
df_test = makeupdata(df_test)
df, df_test = rescaleData(df, df_test)

#print(df)





"""
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#reg = LogisticRegression(random_state=None)
reg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

scores = cross_validation.cross_val_score(
    reg,
    X, #df[predictors],
    train_survival, #df["Survived"],
    cv=3
)

print(scores.mean())
"""

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name"]

# Perform the logistic regression
logistic = LogisticRegression(max_iter = 200, random_state = None)
logistic.fit(df[predictors],df["Survived"])
#print(logistic.coef_)

scores = cross_validation.cross_val_score(
    logistic,
    df[predictors],
    df["Survived"],
    cv=3
)

print("scores for logistic regression: ", scores.mean())


# Perform the random forest classification
rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
rf.fit(df[predictors],df["Survived"])

scores = cross_validation.cross_val_score(
    rf,
    df[predictors],
    df["Survived"],
    cv=3
)

print("scores for random forest: ", scores.mean())


# Perform the SVM classification
svmclassifier = svm.SVC(kernel='rbf', max_iter=-1, random_state=None)
svmclassifier.fit(df[predictors],df["Survived"])

scores = cross_validation.cross_val_score(
    svmclassifier,
    df[predictors],
    df["Survived"],
    cv=3
)

print("scores for SVM: ", scores.mean())


#NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
#NN.fit(df[predictors],df["Survived"])

#scores = cross_validation.cross_val_score(
#    NN,
#    df[predictors],
#    df["Survived"],
#    cv=3
#)

#print("scores for NN: ", scores.mean())


print 'Predicting...'

# perform the predictions
predictionLogistic = logistic.predict(df_test[predictors])
predictionRandomForest = rf.predict(df_test[predictors])
predictionSVM = svmclassifier.predict(df_test[predictors])
#predictionNN = NN.predict(df_test[predictors])

# make an average prediction
averagePrediction = 1./3*(predictionLogistic + predictionRandomForest + predictionSVM)

averagePrediction[np.where(averagePrediction<=0.1)] = 0
averagePrediction[np.where(averagePrediction>0.9)] = 1

print(np.where(averagePrediction == 0))
print(np.where(averagePrediction == 1))

# provides 0.77990
#averagePrediction[np.where(averagePrediction == 1./3)] = 1
#averagePrediction[np.where(averagePrediction == 2./3)] = 0


# provides 0.77990
averagePrediction[np.where(averagePrediction == 1./3)] = 0
averagePrediction[np.where(averagePrediction == 2./3)] = 1

# but changes 52 values of side...

# write down the general predictions in a csv file to check out
ids = df_test['PassengerId'].values
generalpredictions_file = open("myGeneralPredictions.csv", "wb")
open_file_object = csv.writer(generalpredictions_file)
open_file_object.writerow(["PassengerId","SurvivedLogistic","SurvivedRandomForest","SurvivedSVM" ])
open_file_object.writerows(zip(ids, predictionLogistic, predictionRandomForest, predictionSVM))
generalpredictions_file.close()

# write down the final predictions in a csv file as needed by kaggle
generalpredictions_file = open("myFinalPredictions.csv", "wb")
open_file_object = csv.writer(generalpredictions_file)
open_file_object.writerow(["PassengerId","Survived" ])
open_file_object.writerows(zip(ids, averagePrediction))
generalpredictions_file.close()

print 'Done.'

#put data to x and y vectors
#xWeight = df.iloc[:,0].values
#xWeight = xWeight.reshape(len(xWeight),1)
