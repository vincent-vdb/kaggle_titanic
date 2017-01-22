import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression


# get the training data
df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

#get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe

#put it all together
fulldf = pd.concat([df, df_test], axis = 0)


# all sex info are correctly filled up, change them to 0 and 1
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# some ages are not given, give them the mean age
mean_age = df['Age'].dropna().mean()
df.Age[df.Age.isnull()] = mean_age



train_sex = df['Sex'].values
train_survival = df['Survived'].values
train_class = df['Pclass'].values
train_parents = df['Parch'].values
train_siblings = df['SibSp'].values

train_embarked = df['Embarked'].values

# get the percentage of survived women
numberOfWomen = np.size(np.where(train_sex == 0))

survival_women = train_survival[np.where( train_sex == 0)]
numberOfSurvivedWomen = np.size(np.where(survival_women == 1))


PercentageSurvivedWomen = 1.*numberOfSurvivedWomen/numberOfWomen

print("percentage of women who survived: ", PercentageSurvivedWomen)

# get the percentage of survived men
numberOfMen = np.size(np.where(train_sex == 1))

survival_men = train_survival[np.where( train_sex == 1)]
numberOfSurvivedMen = np.size(np.where(survival_men == 1))


PercentageSurvivedMen = 1.*numberOfSurvivedMen/numberOfMen

print("percentage of men who survived: ", PercentageSurvivedMen)


# get the percentage of the different classes who survived
numberOfClass1 = np.size(np.where(train_class == 1))
numberOfClass2 = np.size(np.where(train_class == 2))
numberOfClass3 = np.size(np.where(train_class == 3))

survival_class1 = train_survival[np.where(train_class == 1)]
numberOfSurvivorClass1 = np.size(np.where(survival_class1 == 1))

survival_class2 = train_survival[np.where(train_class == 2)]
numberOfSurvivorClass2 = np.size(np.where(survival_class2 == 1))

survival_class3 = train_survival[np.where(train_class == 3)]
numberOfSurvivorClass3 = np.size(np.where(survival_class3 == 1))

percentageSurvivedClass1 = 1.*numberOfSurvivorClass1/numberOfClass1
percentageSurvivedClass2 = 1.*numberOfSurvivorClass2/numberOfClass2
percentageSurvivedClass3 = 1.*numberOfSurvivorClass3/numberOfClass3

print("percentage who survived in first class: ", percentageSurvivedClass1)
print("percentage who survived in second class: ", percentageSurvivedClass2)
print("percentage who survived in third class: ", percentageSurvivedClass3)



# get the percentage with parents who survived
numberWithParents = np.size(np.where(train_parents >= 1))
survival_withParents = train_survival[np.where(train_parents >= 1)]
numberOfSurvivorWithParents = np.size(np.where(survival_withParents == 1))

percentageSurvivedWithParents = 1.*numberOfSurvivorWithParents/numberWithParents

# get the percentage without parents who survived
numberWithoutParents = np.size(np.where(train_parents == 0))
survival_withoutParents = train_survival[np.where(train_parents == 0)]
numberOfSurvivorWithoutParents = np.size(np.where(survival_withoutParents == 1))

percentageSurvivedWithoutParents = 1.*numberOfSurvivorWithoutParents/numberWithoutParents

# get the percentage with siblings who survived
numberWithSiblings = np.size(np.where(train_siblings >= 1))
survival_withSiblings = train_survival[np.where(train_siblings >= 1)]
numberOfSurvivorWithSiblings = np.size(np.where(survival_withSiblings == 1))

percentageSurvivedWithSiblings = 1.*numberOfSurvivorWithSiblings/numberWithSiblings

# get the percentage without siblings who survived
numberWithoutSiblings = np.size(np.where(train_siblings == 0))
survival_withoutSiblings = train_survival[np.where(train_siblings == 0)]
numberOfSurvivorWithoutSiblings = np.size(np.where(survival_withoutSiblings == 1))

percentageSurvivedWithoutSiblings = 1.*numberOfSurvivorWithoutSiblings/numberWithoutSiblings

print("percentage who survived with parents: ", percentageSurvivedWithParents)
print("percentage who survived without parents: ", percentageSurvivedWithoutParents)
print("percentage who survived with siblings: ", percentageSurvivedWithSiblings)
print("percentage who survived without siblings: ", percentageSurvivedWithoutSiblings)



# get the percentage embarked from C who survived
numberFromC = np.size(np.where(train_embarked == 'C'))
survival_FromC = train_survival[np.where(train_embarked == 'C')]
numberOfSurvivorFromC = np.size(np.where(survival_FromC == 1))

percentageSurvivedFromC = 1.*numberOfSurvivorFromC/numberFromC

print("percentage who survived From Cherbourg: ", percentageSurvivedFromC)

# get the percentage embarked from S who survived
numberFromS = np.size(np.where(train_embarked == 'S'))
survival_FromS = train_survival[np.where(train_embarked == 'S')]
numberOfSurvivorFromS = np.size(np.where(survival_FromS == 1))

percentageSurvivedFromS = 1.*numberOfSurvivorFromS/numberFromS

print("percentage who survived From Southampton: ", percentageSurvivedFromS)

# get the percentage embarked from Q who survived
numberFromQ = np.size(np.where(train_embarked == 'Q'))
survival_FromQ = train_survival[np.where(train_embarked == 'Q')]
numberOfSurvivorFromQ = np.size(np.where(survival_FromQ == 1))

percentageSurvivedFromQ = 1.*numberOfSurvivorFromQ/numberFromQ

print("percentage who survived From Queenstown: ", percentageSurvivedFromQ)



### play with the names, WIP
# map the titles as Mister : 0, Miss : 1, Mistress : 2, Others : 3
#df.Name[0].split()[1]
title = []
for name in df.Name:
  tmp = name.split(',')[1].split()[0]
  if tmp == 'Mr.' :
    title.append(0)
  elif tmp == 'Miss.':
    title.append(1)
  elif tmp == 'Mrs.' :
    title.append(2)
  else :
    title.append(3)



# plot some data for easy viz
#plt.plot(df['Age'], df['Survived'], 'rx')
#plt.show()


