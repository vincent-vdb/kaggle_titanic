import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier


def makeupdata(df):
  # all sex info are correctly filled up, change them to 0 and 1
  df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  # some ages are not given, give them the mean age
  mean_age = df['Age'].dropna().mean()
  df.Age[df.Age.isnull()] = mean_age

  # make unknown embarked from Cherbourg
  df.Embarked[df.Embarked.isnull()] = 'C'

  # convert embarked info to int
  df['Embarked'] = df['Embarked'].map({'S':-1, 'Q':0, 'C':1}).astype(int)

  #rescale the age to the mean with unit variance
  mean_age = np.mean(df['Age'])
  var_age = np.var(df['Age'])
  df['Age'] = 1.*(df['Age'] - mean_age)/var_age

  #rescale the age to the mean with unit variance
  mean_fare = np.mean(df['Fare'])
  var_fare = np.var(df['Fare'])
  df['Fare'] = 1.*(df['Fare']-mean_fare)/var_fare

  #put siblings to 0s or 1s
#  df['SibSp'][np.where(df['SibSp'] > 1)] = 1

  #put parents to 0s or 1s
#  df['Parch'][np.where(df['SibSp'] > 1)] = 1

  ### play with the names to get the titles
  # map the titles as Mister : 0, Mistress : 1, Miss : 2, Others : 3
  title = []
  for name in df.Name:
    tmp = name.split(',')[1].split()[0]
    if tmp == 'Mr.' :
      title.append(0)
    elif tmp == 'Mrs.':
      title.append(1)
    elif tmp == 'Miss.' :
      title.append(2)
    else :
      title.append(3)

  return df

def rescaleData (df_train, df_test):
  
  # all sex info are correctly filled up, change them to 0 and 1
  fullDf = pd.concat([df_train, df_test], axis = 0)
  fullDf['Sex'] = fullDf['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  # some ages are not given, give them the mean age
  mean_age = fullDf['Age'].dropna().mean()
  fullDf.Age[fullDf.Age.isnull()] = mean_age

  # make unknown embarked from Cherbourg
  fullDf.Embarked[fullDf.Embarked.isnull()] = 'C'

  # convert embarked info to int
  fullDf['Embarked'] = fullDf['Embarked'].map({'S':-1, 'Q':0, 'C':1}).astype(int)

  #rescale the age to the mean with unit variance
  mean_age = np.mean(fullDf['Age'])
  var_age = np.var(fullDf['Age'])
  fullDf['Age'] = 1.*(fullDf['Age'] - mean_age)/var_age

  #rescale the age to the mean with unit variance
  mean_fare = np.mean(fullDf['Fare'])
  var_fare = np.var(fullDf['Fare'])
  fullDf['Fare'] = 1.*(fullDf['Fare']-mean_fare)/var_fare

  #put siblings to 0s or 1s
#  fullDf['SibSp'][np.where(fullDf['SibSp'] > 1)] = 1

  #put parents to 0s or 1s
#  fullDf['Parch'][np.where(fullDf['SibSp'] > 1)] = 1

  ### play with the names to get the titles
  # map the titles as Mister : 0, Mistress : 1, Miss : 2, Others : 3
  title = []
  for name in fullDf.Name:
    tmp = name.split(',')[1].split()[0]
    if tmp == 'Mr.' :
      title.append(0)
    elif tmp == 'Mrs.':
      title.append(1)
    elif tmp == 'Miss.' :
      title.append(2)
    else :
      title.append(3)





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


# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","Name"]


df = makeupdata(df)


print(df)




# Concatenate all relevant data in an X array
#X = np.stack((train_sex, train_class, train_age, train_embarked, train_parents, train_siblings, train_fare, title), axis = 1)

# Perform the logistic regression
logistic = LogisticRegression()
logistic.fit(X,train_survival)

print(logistic.coef_)






ids = df_test['PassengerId'].values


# get the relevant info as the training data
Xtest = np.stack((test_sex, test_class, test_age, test_embarked, test_parents, test_siblings, test_fare, title), axis = 1)

print 'Predicting...'

# perform the prediction
output = logistic.predict(Xtest)

# write down the prediction in a csv file as needed by kaggle
predictions_file = open("myLogisticRegression.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

#put data to x and y vectors
#xWeight = df.iloc[:,0].values
#xWeight = xWeight.reshape(len(xWeight),1)
