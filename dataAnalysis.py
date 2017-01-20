import pandas as pd
import numpy as np
#import csv as csv
import matplotlib.pyplot as plt


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

# get the percentage of survived women
numberOfWomen = np.size(np.where(train_sex == 0))

survival_women = train_survival[np.where( train_sex == 0)]
numberOfSurvivedWomen = np.size(np.where(survival_women == 1))


PercentageSurvivedWomen = 1.*numberOfSurvivedWomen/numberOfWomen

print("percentage of women who survived: ",PercentageSurvivedWomen)

# get the percentage of survived men
numberOfMen = np.size(np.where(train_sex == 1))

survival_men = train_survival[np.where( train_sex == 1)]
numberOfSurvivedMen = np.size(np.where(survival_men == 1))


PercentageSurvivedMen = 1.*numberOfSurvivedMen/numberOfMen

print("percentage of men who survived: ",PercentageSurvivedMen)


# get the percentage of 1st class who survived
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

print("percentage who survived in first class: ",percentageSurvivedClass1)
print("percentage who survived in second class: ",percentageSurvivedClass2)
print("percentage who survived in third class: ",percentageSurvivedClass3)
# plot some data for easy viz
#plt.plot(df['Age'], df['Survived'], 'rx')
#plt.show()


# get the test data
df_test = pd.read_csv('test.csv', header=0)        # Load the train file into a dataframe


# change the sex info same way as in the train data
df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


#put data to x and y vectors
#xWeight = df.iloc[:,0].values
#xWeight = xWeight.reshape(len(xWeight),1)
