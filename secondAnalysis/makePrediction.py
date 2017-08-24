import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# get the training data
df = pd.read_csv('../train.csv', header=0)        # Load the train file into a dataframe
#get the test data
df_test = pd.read_csv('../test.csv', header=0)        # Load the train file into a dataframe

#put it all together
fulldf = pd.concat([df, df_test], axis = 0)
"""
      fulldf.count()
Age            1046
Cabin           295
Embarked       1307
Fare           1308
Name           1309
Parch          1309
PassengerId    1309
Pclass         1309
Sex            1309
SibSp          1309
Survived        891
Ticket         1309

  df.count()
PassengerId    891
Survived       891
Pclass         891
Name           891
Sex            891
Age            714
SibSp          891
Parch          891
Ticket         891
Fare           891
Cabin          204
Embarked       889

"""

def parseData(X):
  target = X.Survived

  X.Fare = scale(X.Fare)

  features = X[['Parch', 'SibSp', 'Fare']]

  pclass = pd.get_dummies(X['Pclass'], prefix='splitClass')
  features = features.join(pclass)

  psex = pd.get_dummies(X['Sex'], prefix='splitSex') 
  features = features.join(psex)

  child = X['Age'] < 8
  features = features.join(child)

#  tmpAge = X['Age']
#  meanAge = X['Age'].mean()
#  tmpAge[tmpAge.isnull()] = meanAge
#  features = features.join(X['Age'])

  tmp = X['Embarked']
  tmp[tmp.isnull()] = 'C'
  pembarked = pd.get_dummies(X['Embarked'], prefix='splitEmbarked') 
  features = features.join(pembarked)

  title = df.Name.map(lambda x: x.split(',')[1].split('.')[0])
  ptitle = pd.get_dummies(title, prefix='title')
  features = features.join(ptitle)

  surname = df.Name.map(lambda x: '(' in x)
  features = features.join(surname)

  cabin = df.Cabin.map(lambda x: x[0] if not pd.isnull(x) else -1)
  pcabin = pd.get_dummies(cabin, prefix='cabin')
  features = features.join(pcabin)

  return features, target

def computeScore(clf, X, y):
  xval = cross_val_score(clf, X, y, cv=5)
  return xval

X, y = parseData(df)
lr = LogisticRegression()
print("logistic regression: ",np.mean(computeScore(lr, X, y)))
rdf = RandomForestClassifier()
print("random forest classification: ",np.mean(computeScore(rdf, X, y)))
grad = GradientBoostingClassifier()
print("gradient boosting: ",np.mean(computeScore(grad, X, y))) 
mySVC = SVC()
print("SVM Classifier: ", np.mean(computeScore(mySVC, X, y)))
knn = KNeighborsClassifier()
print("k-NN Classifier: ", np.mean(computeScore(knn, X, y)))

