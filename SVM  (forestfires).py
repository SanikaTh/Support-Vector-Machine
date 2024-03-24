# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:45:40 2024

@author: HP
"""
'''
A construction firm wants to develop a suburban locality with new infrastructure but they
 might incur losses if they cannot sell the properties. To overcome this, they consult an 
 analytics firm to get insights on how densely the area is populated and the income levels
 of residents. Use the Support Vector Machines algorithm on the given dataset and draw out
 insights and also comment on the viability of investing in that area.
 
 Business Objectives:
Minimize Risk: Assess the potential market demand for the developed properties to avoid losses.
Maximize Profit: Sell properties at optimal prices by understanding the demographics and income levels of potential buyers.
Strategic Investment: Make informed decisions about investing in infrastructure development based on population density and income levels.

Constraints:
Data Quality: Ensure the dataset is reliable and representative of the target population.
Model Interpretability: Interpret the SVM model's results to draw meaningful insights for decision-making.
Resource Allocation: Allocate resources efficiently based on the predicted demand and market potential.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

forest=pd.read_csv("D:/Documents/Datasets/forestfires.csv")
forest.head()
'''
month  day  FFMC   DMC  ...  monthnov  monthoct  monthsep  size_category
0   mar  fri  86.2  26.2  ...         0         0         0          small
1   oct  tue  90.6  35.4  ...         0         1         0          small
2   oct  sat  90.6  43.7  ...         0         1         0          small
3   mar  fri  91.7  33.3  ...         0         0         0          small
4   mar  sun  89.3  51.3  ...         0         0         0          small

[5 rows x 31 columns]

'''
forest.shape
#rows=517 and columns=31

forest.columns
'''
Index(['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
       'rain', 'area', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
       'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
       'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
       'monthoct', 'monthsep', 'size_category'],
      dtype='object')
'''

forest.dtypes
'''
month             object
day               object
FFMC             float64
DMC              float64
DC               float64
ISI              float64
temp             float64
RH                 int64
wind             float64
rain             float64
area             float64
dayfri             int64
daymon             int64
daysat             int64
daysun             int64
daythu             int64
daytue             int64
daywed             int64
monthapr           int64
monthaug           int64
monthdec           int64
monthfeb           int64
monthjan           int64
monthjul           int64
monthjun           int64
monthmar           int64
monthmay           int64
monthnov           int64
monthoct           int64
monthsep           int64
size_category     object
dtype: object
'''
forest.info()

forest.isnull().sum()
#there are no null values

forest.describe()

sns.countplot(x=forest["month"])
plt.show()
#Most of time forest fire occured in Augest and September month

sns.countplot(x=forest["day"])
plt.show()
#Most of time forest fire occured on Friday,Sunday and Saturday.

sns.distplot(forest.FFMC)
plt.show()
#data is normal and slight left skewed
sns.distplot(forest.DMC)
plt.show()
#data is normal and slight right skewed

sns.distplot(forest.DC)
plt.show()
#data is bimodal and normal and slight left skewed

sns.distplot(forest.ISI)
plt.show()
#data is normal and slight RIGHT skewed

sns.distplot(forest.temp)
plt.show()
#data is normal and slight left skewed

sns.distplot(forest.RH)
plt.show()
#data is normal and slight RIGHT skewed

sns.distplot(forest.wind)
plt.show()
#data is normal and slight right skewed

sns.distplot(forest.rain)
plt.show()
#data is normal and slight right skewed

sns.distplot(forest.area)
plt.show()
#data is normal and slight right skewed

sns.boxplot(forest.FFMC)
#There are several outliers

sns.boxplot(forest.DMC)
#There are several outliers

sns.boxplot(forest.DC)
#There are several outliers

sns.boxplot(forest.ISI)
#There are several outliers

sns.boxplot(forest.temp)
#There are several outliers

sns.boxplot(forest.RH)
#There are several outliers

sns.boxplot(forest.wind)
#There are several outliers

sns.boxplot(forest.rain)
#There are several outliers

sns.boxplot(forest.area)
#There are several outliers

#Now let us check the Highest Fire In KM?
highest_fire_area = forest.sort_values(by="area", ascending=False).head(50)
highest_fire_area.head(5)

plt.figure(figsize=(8, 6))

plt.title("Temperature vs area of fire" )
plt.bar(highest_fire_area['temp'], highest_fire_area['area'])

plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()
#once the fire starts,almost 1000+ sq area's temperature goes beyond 25 in Celcius and 
#around 750km area is facing temp 30+ celcius

#Now let us check the Highest rain In the forest?
highest_rain=forest.sort_values(by="rain",ascending=False)[['month', 'day', 'rain']].head(5)
highest_rain
#highest rain observed in the month of aug

#Let us check highest and lowest temperature in month and day wise
highest_temp = forest.sort_values(by='temp', ascending=False)[['month', 'day', 'temp']].head(5)

lowest_temp =  forest.sort_values(by='temp', ascending=True)[['month', 'day', 'temp']].head(5)


print("Highest temp:\n",highest_temp)
#In august month, tempeture is highest
print("-"*50)
print("Lowest temp:\n",lowest_temp)
#In december month, temperatur is lowest

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.size_category=labelencoder.fit_transform(forest.size_category)

#month and day columns are already encoded with one hot encoder
#so we can drop month and day columns from dataset
forest1=forest.drop(["month","day"],axis=1)

forest1.head()

#remove the outliers
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="FFMC",tail="both")
forest1["FFMC"]=winsor.fit_transform(forest1[["FFMC"]])
sns.boxplot(forest1["FFMC"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="DC",tail="both")
forest["DC"]=winsor.fit_transform(forest[["DC"]])
sns.boxplot(forest["DC"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="ISI",tail="both")
forest1["ISI"]=winsor.fit_transform(forest1[["ISI"]])
sns.boxplot(forest1["ISI"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="RH",tail="both")
forest1["RH"]=winsor.fit_transform(forest1[["RH"]])
sns.boxplot(forest1["RH"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="wind",tail="both")
forest1["wind"]=winsor.fit_transform(forest1[["wind"]])
sns.boxplot(forest1["wind"])

winsor=Winsorizer(capping_method="gaussian",fold=1.5,variables="rain",tail="both")
forest1["rain"]=winsor.fit_transform(forest1[["rain"]])
sns.boxplot(forest1["rain"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="area",tail="both")
forest1["area"]=winsor.fit_transform(forest1[["area"]])
sns.boxplot(forest1["area"])

plt.figure(figsize=(20,20))
tc=forest1.corr()
sns.heatmap(tc,cmap="YlGnBu",annot=True,fmt=".2f")
plt.show()
#all the variables are moderately correlated with size_category except area

x=forest1.drop("size_category",axis=1)
y=forest1["size_category"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.svm import SVC
model=SVC(kernel="linear")
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred

np.mean(y_pred==y_test)
#0.9807692307692307
