# -*- coding: utf-8 -*

"""
Created on Wed Mar 27 16:13:34 2019

@author: Ralph

*
the data is the "Heart Disease UCI" dataset, obtained from Kaggle. Thank you
to user 'ronitf' for providing the data to kaggle. 

The goal of this project is to be able to determine the most relevant features
of the data, in terms of what most likely affects the probability of having
heart disease


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir('C:\\Users\\Ralph\\Desktop\\Python\\Projects')
original_data = pd.read_csv('heart.csv')

"""
column definintions:
    
age: The person's age in years

sex: The person's sex (1 = male, 0 = female)

cp: The chest pain experienced (Value 1: typical angina,
 Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

trestbps: The person's resting blood pressure (mm Hg on admission)

chol: The person's cholesterol measurement in mg/dl

fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

restecg: Resting electrocardiographic measurement (0 = normal, 
1 = having ST-T wave abnormality, 
2 = showing probable or definite left ventricular hypertrophy by Estes' 
criteria)

thalach: The person's maximum heart rate achieved

exang: Exercise induced angina (1 = yes; 0 = no)

oldpeak: ST depression induced by exercise relative to rest
('ST' relates to positions on the ECG plot. See more here)

slope: the slope of the peak exercise ST segment (Value 1: upsloping,
Value 2: flat, Value 3: downsloping)

ca: The number of major vessels (0-3)

thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect;
 7 = reversable defect)

target: Heart disease (0 = no, 1 = yes)

"""

wdata = original_data[0:]
wdata.columns
wdata.columns = ['age','sex','chestPainType','restingBP','serumCholesterol',
                 'fastingBloodSugar>120','restingEKG','maxHeartRate',
                 'exerciseInducedAngina','STDepression','STSlope',
                 'majorVesselCount','thalassemia','target']
wdata.dtypes

#categorizing the data appropriately:
#sex:
wdata['sex'][wdata['sex'] == 1] = 'male'
wdata['sex'][wdata['sex'] == 0] = 'female'

#chest pain type:
wdata['chestPainType'][wdata['chestPainType'] == 0] = None
wdata['chestPainType'][wdata['chestPainType'] == 1] = 'typical angina'
wdata['chestPainType'][wdata['chestPainType'] == 2] = 'atypical angina'
wdata['chestPainType'][wdata['chestPainType'] == 3] = 'non-anginal pain'
wdata['chestPainType'][wdata['chestPainType'] == 4] = 'asymptomatic'

#fasting blood sugar > 120
wdata['fastingBloodSugar>120'][wdata['fastingBloodSugar>120'] == 1] = 'True'
wdata['fastingBloodSugar>120'][wdata['fastingBloodSugar>120'] == 0] = 'False'

#resting EKG
wdata['restingEKG'][wdata['restingEKG'] == 0] = 'normal'
wdata['restingEKG'][wdata['restingEKG'] == 1] = 'ST-T wave abnormality'
wdata['restingEKG'][wdata['restingEKG'] == 2] = "probable or definite left ventricular hypertrophy by Estes' criteria"

#exercise induced angina:
wdata['exerciseInducedAngina'][wdata['exerciseInducedAngina'] == 1] = 'yes'
wdata['exerciseInducedAngina'][wdata['exerciseInducedAngina'] == 0] = 'no'

#ST slope:
wdata['STSlope'][wdata['STSlope'] == 0] = None
wdata['STSlope'][wdata['STSlope'] == 1] = 'upsloping'
wdata['STSlope'][wdata['STSlope'] == 2] = 'flat'
wdata['STSlope'][wdata['STSlope'] == 3] = 'downsloping'

#thalassemia:
wdata['thalassemia'][wdata['thalassemia'] == 0] = None
wdata['thalassemia'][wdata['thalassemia'] == 1] = 'normal'
wdata['thalassemia'][wdata['thalassemia'] == 2] = 'fixed defect'
wdata['thalassemia'][wdata['thalassemia'] == 3] = 'reversible defect'

#we will set the none vals as str nones for now, and drop them after LE/OHE


#set the columns above as categorical types:

wdata['sex'] = wdata['sex'].astype('object')
wdata['chestPainType'] = wdata['chestPainType'].astype('object')
wdata['fastingBloodSugar>120'] = wdata['fastingBloodSugar>120'].astype('object')
wdata['restingEKG'] = wdata['restingEKG'].astype('object')
wdata['exerciseInducedAngina'] = wdata['exerciseInducedAngina'].astype('object')
wdata['STSlope'] = wdata['STSlope'].astype('object')
wdata['thalassemia'] = wdata['thalassemia'].astype('object')

#missing value ratio:
(pd.DataFrame(wdata).isnull().sum()/len(wdata))*100

#we'll drop chestPainType since it's missing 47% of it's columns

X = wdata[['age','sex','restingBP','serumCholesterol',
                 'fastingBloodSugar>120','restingEKG','maxHeartRate',
                 'exerciseInducedAngina','STDepression','STSlope',
                 'majorVesselCount','thalassemia']]
#mvr check:
(pd.DataFrame(X).isnull().sum()/len(X))*100
print((pd.DataFrame(X).isnull().sum()/len(X))*100)

X = X[X['STSlope']!= "None"]
y = wdata.loc[:,['target']].values

#get dummies:
X = pd.get_dummies(X, drop_first = True)


#quickly view a correlation heatmap of the data:
plt.figure(figsize=(20,10))
sns.heatmap(X.corr(), annot=True)
plt.show()



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train.ravel(), cv = 10)
accmean = accuracies.mean()
stddev = accuracies.std()

print("Accuracy is "+ str(round(accmean*100, 3)) +"%")

#feature importance:
#feature selection via random forest:
X.columns
Xfi = X[['age', 'restingBP', 'serumCholesterol', 'maxHeartRate', 'STDepression',
       'majorVesselCount', 'sex_male', 'fastingBloodSugar>120_True',
       'restingEKG_normal',
       "restingEKG_probable or definite left ventricular hypertrophy by Estes' criteria",
       'exerciseInducedAngina_yes', 'STSlope_upsloping', 'thalassemia_normal',
       'thalassemia_reversible defect']]
XColNames = X.columns.values.tolist()
Xfi.columns = XColNames[0:]
features = Xfi.columns
importances = classifier.feature_importances_
indices = np.argsort(importances)

plt.title('Heart Disease Feature Importances')
plt.barh(range(len(indices)),importances[indices], color = 'r',
         align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
print("Feature importances defined by a model wth accucracy of" + str(round(accmean*100,3)) + "%")


##############
##############
##############

#model refinement using back propagation and grid search CV:



plt.figure(figsize=(18,9))
sns.heatmap(X.corr(), annot=True)
plt.show()

corr_grid = pd.DataFrame(X.corr())

"""
this section will be frequently updated, as better models arise
"""


rX = X[['age', 'restingBP', 'serumCholesterol', 'maxHeartRate', 'STDepression',
       'majorVesselCount', 'sex_male', 'fastingBloodSugar>120_True',
       'restingEKG_normal',
       "restingEKG_probable or definite left ventricular hypertrophy by Estes' criteria",
       'exerciseInducedAngina_yes', 'STSlope_upsloping', 'thalassemia_normal',
       'thalassemia_reversible defect']]

ls = ['age', 'restingBP', 'serumCholesterol', 'maxHeartRate', 'STDepression',
       'majorVesselCount', 'sex_male', 'fastingBloodSugar>120_True',
       'restingEKG_normal',
       "restingEKG_probable or definite left ventricular hypertrophy by Estes' criteria",
       'exerciseInducedAngina_yes', 'STSlope_upsloping', 'thalassemia_normal',
       'thalassemia_reversible defect']
ry = y[0:]   

hardSearchBlock = []
for i in range(0,len(rX.columns)):

    ls = ['age', 'restingBP', 'serumCholesterol', 'maxHeartRate', 'STDepression',
           'majorVesselCount', 'sex_male', 'fastingBloodSugar>120_True',
           'restingEKG_normal',
           "restingEKG_probable or definite left ventricular hypertrophy by Estes' criteria",
           'exerciseInducedAngina_yes', 'STSlope_upsloping', 'thalassemia_normal',
           'thalassemia_reversible defect']
    
    ls.remove(ls[i])
    rX = X[ls]
    rX_train, rX_test, ry_train, ry_test = train_test_split(rX, ry,
                                                            test_size = 0.20,
                                                            random_state = 0)
    
    rsc = StandardScaler()
    rX_train = rsc.fit_transform(rX_train)
    rX_test = rsc.transform(rX_test)
    
    rclassifier = XGBClassifier(booster = 'gbtree', 
                                colsample_bytree = 0.5, 
                                eta = 0.01, gamma = 10, max_depth = 3,
                                subsample = 0.5)
    rclassifier.fit(rX_train, ry_train.ravel())
    
    # Predicting the Test set results
    ry_pred = rclassifier.predict(rX_test)
    
    # Making the Confusion Matrix
    rcm = confusion_matrix(ry_test, ry_pred)
    rcm
    def rcm_calc():
        rcmc = ((rcm[0][0]+rcm[1][1])/(rcm[0][0]+rcm[1][1]+rcm[1][0]+rcm[0][1]))*100
        print("Accucracy is " + str(round(rcmc,3)) + "%")
    
    raccuracies = cross_val_score(estimator = rclassifier, X = rX_train,
                                  y = ry_train.ravel(), cv = 10)
    raccmean = raccuracies.mean()
    rstddev = raccuracies.std()
    print ("Accucracy is " + str(round(raccmean,3)) + "%")
    
    #feature importance and graphing of the feature importance:
    rXfi = rX[0:]
    rfeatures = rXfi.columns
    rimportances = rclassifier.feature_importances_
    rindices = np.argsort(rimportances)
    
    plt.title('Heart Disease Feature Importances')
    plt.barh(range(len(rindices)),rimportances[rindices], color = 'b',
             align = 'center')
    plt.yticks(range(len(rindices)), [rfeatures[i] for i in rindices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    print(accmean, raccmean, ((raccmean/accmean)*100)-100)
    hardSearchBlock.append(raccmean)

idx = list(range(0,len(hardSearchBlock)))

HBSGrid = pd.DataFrame({'HSBVals': hardSearchBlock, 'rX[index]': idx})
HBSGrid['rX[index]']=HBSGrid['rX[index]'].astype(object)
HBSGrid = HBSGrid.sort_values(by ='HSBVals' ,ascending = False)

#PARAMETER TUNING OF XGBOOST:
from sklearn.model_selection import GridSearchCV

"""
#blocked off so as not to rerun grid search (~2hours) on initialization.
#uncomment to allow for grid search.

parameters = [{'tree_method': ['gpu_hist'],
               'booster':['gbtree','gblinear','dart'],
               'gamma':[0,0.1,1,10,100,1000],
               'eta':[0.01,0.05,0.1,0.15,0.2,0.25,0.3],
               'max_depth':[3,4,5,6,7,8,9,10],
               'subsample':[0.5,0.6,0.7,0.8,0.9,1.0],
               'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1.0]}]
    
grid_search = GridSearchCV(estimator = rclassifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1, iid=True)
grid_search = grid_search.fit(rX_train, ry_train.ravel())
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""


#CURRENTLY NEEDS MODEL OPTIMIZATION.




