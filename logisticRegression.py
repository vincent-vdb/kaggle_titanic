import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression


# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe


# all sex info are correctly filled up, change them to 0 and 1
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# some ages are not given, give them the mean age
mean_age = df['Age'].dropna().mean()
df.Age[df.Age.isnull()] = mean_age



train_sex = df['Sex'].values
train_survival = df['Survived'].values
train_class = df['Pclass'].values



#plt.plot(df['Age'], df['Survived'], 'rx')
#plt.show()

#X = np.concatenate((train_sex, train_class), axis=1)

X = np.stack((train_sex, train_class), axis = 1)

logistic = LogisticRegression()
logistic.fit(X,train_survival)




# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe
ids = df_test['PassengerId'].values

# change the sex info same way as in the train data
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test_sex = df_test['Sex'].values
test_class = df_test['Pclass'].values


Xtest = np.stack((test_sex, test_class), axis = 1)

print 'Predicting...'

output = logistic.predict(Xtest)

predictions_file = open("myfirstLogisticRegression.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

#put data to x and y vectors
#xWeight = df.iloc[:,0].values
#xWeight = xWeight.reshape(len(xWeight),1)
