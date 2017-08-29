import pandas as pd
import numpy as np
import csv as csv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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

fulldf.reset_index(inplace=True)
 
# reset_index() generates a new column that we don't want, so let's get rid of it
fulldf.drop('index', axis=1, inplace=True)
 
# the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'
fulldf = fulldf.reindex_axis(df.columns, axis=1)


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


### Populate missing ages  using RandomForestClassifier
def setMissingAges(df):
    
    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['Age', 'Parch', 'SibSp', 'Pclass']]
    
    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]
    
    # All age values are stored in a target array
    y = knownAge.values[:, 0]
    
    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df

def parseData(df):
  target = df.Survived

  df.Fare[df.Fare.isnull()] = df.Fare.mean()
  df.Fare = scale(df.Fare)

  features = df[['Parch', 'SibSp', 'Fare', 'Age']]

  pclass = pd.get_dummies(df['Pclass'], prefix='splitClass')
  features = features.join(pclass)

  psex = pd.get_dummies(df['Sex'], prefix='splitSex') 
  features = features.join(psex)

  child = df['Age'] < 8
  child.name='Child'
  features = features.join(child)

  tmp = df['Embarked']
  tmp[tmp.isnull()] = 'C'
  pembarked = pd.get_dummies(df['Embarked'], prefix='splitEmbarked') 
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

def computeAllScores(df):
  df = setMissingAges(df)
  X, y = parseData(df)
  lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=.1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
  print("logistic regression: ",np.mean(computeScore(lr, X, y)))
  rdf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
  print("random forest classification: ",np.mean(computeScore(rdf, X, y)))
  grad = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10000, subsample=0.9, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
  print("gradient boosting: ",np.mean(computeScore(grad, X, y))) 
  mySVC = SVC(C=.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
  print("SVM Classifier: ", np.mean(computeScore(mySVC, X, y)))
  knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
  print("k-NN Classifier: ", np.mean(computeScore(knn, X, y)))


def performPrediction(fulldf, df, df_test):
  fulldf = setMissingAges(fulldf)

  X, y = parseData(fulldf)

  lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=.1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
  lr.fit(X[:df.shape[0]], y[:df.shape[0]])
  lrPrediction = lr.predict(X[df.shape[0]:])

  rdf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
  rdf.fit(X[:df.shape[0]], y[:df.shape[0]])
  rdfPrediction = rdf.predict(X[df.shape[0]:])

  grad = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10000, subsample=0.9, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
  grad.fit(X[:df.shape[0]], y[:df.shape[0]])
  gradPrediction = grad.predict(X[df.shape[0]:])

  mySVC = SVC(C=.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
  mySVC.fit(X[:df.shape[0]], y[:df.shape[0]])
  mySVCPrediction = mySVC.predict(X[df.shape[0]:])

  knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
  knn.fit(X[:df.shape[0]], y[:df.shape[0]])
  knnPrediction = knn.predict(X[df.shape[0]:])

  # make an average prediction
  averagePrediction = 1./5*(lrPrediction + rdfPrediction + gradPrediction + mySVCPrediction + knnPrediction)

  averagePrediction[np.where(averagePrediction<=0.5)] = 0
  averagePrediction[np.where(averagePrediction>0.5)] = 1

  # write down the general predictions in a csv file to check out
  ids = fulldf['PassengerId'].values[df.shape[0]:]
  generalpredictions_file = open("myGeneralPredictions.csv", "wb")
  open_file_object = csv.writer(generalpredictions_file)
  open_file_object.writerow(["PassengerId","SurvivedLogistic","SurvivedRandomForest", "SurvivedGrad", "SurvivedSVM", "SurvivedkNN"])
  open_file_object.writerows(zip(ids, lrPrediction, rdfPrediction, gradPrediction, mySVCPrediction, knnPrediction))
  generalpredictions_file.close()

  # write down the final predictions in a csv file as needed by kaggle
  generalpredictions_file = open("averagePredictions.csv", "wb")
  open_file_object = csv.writer(generalpredictions_file)
  open_file_object.writerow(["PassengerId", "Survived"])
  open_file_object.writerows(zip(ids, averagePrediction.astype(int)))
  #open_file_object.writerows(zip(ids, averagePrediction.astype(int)))
  generalpredictions_file.close()

computeAllScores(df)
performPrediction(fulldf, df, df_test)

