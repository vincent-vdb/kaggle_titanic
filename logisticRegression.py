import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier


# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# all sex info are correctly filled up, change them to 0 and 1
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# some ages are not given, give them the mean age
train_mean_age = df['Age'].dropna().mean()
df.Age[df.Age.isnull()] = train_mean_age

# make unknown embarked the most common
df.Embarked[df.Embarked.isnull()] = 'S'

# convert embarked info to int
df['Embarked'] = df['Embarked'].map({'S':-1, 'Q':0, 'C':1}).astype(int)


train_sex = df['Sex'].values
train_class = df['Pclass'].values
train_age = df['Age'].values
train_embarked = df['Embarked'].values
train_fare = df['Fare'].values
train_parents = df['Parch'].values
train_siblings = df['SibSp'].values

#rescale the age to the mean with unit variance
train_mean_age = np.mean(train_age)
train_var_age = np.var(train_age)
train_age = 1.*(train_age - train_mean_age)/train_var_age

#rescale the age to the mean with unit variance
train_mean_fare = np.mean(train_fare)
train_var_fare = np.var(train_fare)
train_fare = 1.*(train_fare-train_mean_fare)/train_var_fare

train_survival = df['Survived'].values

#put siblings to 0s or 1s
train_siblings[np.where(train_siblings >1)] = 1

#put parents to 0s or 1s
train_parents[np.where(train_parents >1)] = 1


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


# Concatenate all relevant data in an X array
X = np.stack((train_sex, train_class, train_age, train_embarked, train_parents, train_siblings, train_fare, title), axis = 1)

# Perform the logistic regression
logistic = LogisticRegression()
logistic.fit(X,train_survival)

print(logistic.coef_)





predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

"""
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



# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe
ids = df_test['PassengerId'].values

# change the sex info same way as in the train data
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# some ages are not given, give them the mean age as for the training data
mean_age = df_test['Age'].dropna().mean()
df_test.Age[df_test.Age.isnull()] = mean_age

# make unknown embarked the most common
df_test.Embarked[df_test.Embarked.isnull()] = 'S'

# convert embarked info to int
df_test['Embarked'] = df_test['Embarked'].map({'S':-1, 'Q':0, 'C':1}).astype(int)

# put non existing fare to mean value
mean_fare = df_test['Fare'].dropna().mean()
df_test.Fare[df_test.Fare.isnull()] = mean_fare

test_sex = df_test['Sex'].values
test_class = df_test['Pclass'].values
test_age = df_test['Age'].values
test_embarked = df_test['Embarked'].values
test_fare = df_test['Fare'].values
test_parents = df_test['Parch'].values
test_siblings = df_test['SibSp'].values

#rescale the same way the test age
test_age = 1.*(test_age - train_mean_age)/train_var_age

#rescale the age to the mean with unit variance
test_fare = 1.*(test_fare-train_mean_fare)/train_var_fare

#put siblings to 0s or 1s
test_siblings[np.where(test_siblings >1)] = 1

#put parents to 0s or 1s
test_parents[np.where(test_parents >1)] = 1

# map the titles as Mister : 0, Mistress : 1, Miss : 2, Others : 3
title = []
for name in df_test.Name:
  tmp = name.split(',')[1].split()[0]
  if tmp == 'Mr.' :
    title.append(0)
  elif tmp == 'Mrs.':
    title.append(1)
  elif tmp == 'Miss.' :
    title.append(2)
  else :
    title.append(3)

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
