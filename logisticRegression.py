import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

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
df['Embarked'] = df['Embarked'].map({'S':0, 'Q':-1, 'C':1}).astype(int)


train_sex = df['Sex'].values
train_class = df['Pclass'].values
train_age = df['Age'].values
train_embarked = df['Embarked'].values


#rescale the age to the mean with unit variance
train_mean_age = np.mean(train_age)
train_var_age = np.var(train_age)
train_age = 1.*(train_age - train_mean_age)/train_var_age


train_survival = df['Survived'].values

# Concatenate all relevant data in an X array
X = np.stack((train_sex, train_class, train_age, train_embarked), axis = 1)

# Perform the logistic regression
logistic = LogisticRegression()
logistic.fit(X,train_survival)




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
df_test['Embarked'] = df_test['Embarked'].map({'S':0, 'Q':-1, 'C':1}).astype(int)



test_sex = df_test['Sex'].values
test_class = df_test['Pclass'].values
test_age = df_test['Age'].values
test_embarked = df_test['Embarked'].values
#rescale the same way the test age
test_age = 1.*(test_age - train_mean_age)/train_var_age


# get the relevant info as the training data
Xtest = np.stack((test_sex, test_class, test_age, test_embarked), axis = 1)

print 'Predicting...'

# perform the prediction
output = logistic.predict(Xtest)

# write down the prediction in a csv file as needed by kaggle
predictions_file = open("myfirstLogisticRegression.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

#put data to x and y vectors
#xWeight = df.iloc[:,0].values
#xWeight = xWeight.reshape(len(xWeight),1)
