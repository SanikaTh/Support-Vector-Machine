# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:47:30 2024

@author: HP
"""

'''
A construction firm wants to develop a suburban locality with new infrastructure 
but they might incur losses if they cannot sell the properties. To overcome this,
they consult an analytics firm to get insights on how densely the area is populated 
and the income levels of residents. Use the Support Vector Machines algorithm on the 
given dataset and draw out insights and also comment on the viability of investing in that area.

Business Objectives:
Minimize Risk: Avoid losses by assessing the potential market demand for the developed properties.
Maximize Profit: Sell properties at optimal prices by understanding the demographics and income levels of potential buyers.
Strategic Investment: Make informed decisions about investing in infrastructure development based on population density and income levels.

Constraints:
Data Quality: Ensure that the dataset is reliable and representative of the target population.
Model Interpretability: Interpret the SVM model's results to draw meaningful insights for decision-making.
Resource Allocation: Allocate resources efficiently based on the predicted demand and market potential.
'''

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv("D:/Documents/Datasets/SalaryData_Test (1).csv")
data1=pd.read_csv("D:/Documents/Datasets/SalaryData_Train (1).csv")
data
'''
 age      workclass      education  ...  hoursperweek          native  Salary
0       25        Private           11th  ...            40   United-States   <=50K
1       38        Private        HS-grad  ...            50   United-States   <=50K
2       28      Local-gov     Assoc-acdm  ...            40   United-States    >50K
3       44        Private   Some-college  ...            40   United-States    >50K
4       34        Private           10th  ...            30   United-States   <=50K
   ...            ...            ...  ...           ...             ...     ...
15055   33        Private      Bachelors  ...            40   United-States   <=50K
15056   39        Private      Bachelors  ...            36   United-States   <=50K
15057   38        Private      Bachelors  ...            50   United-States   <=50K
15058   44        Private      Bachelors  ...            40   United-States   <=50K
15059   35   Self-emp-inc      Bachelors  ...            60   United-States    >50K

[15060 rows x 14 columns]


'''
data1
'''
 age          workclass  ...          native  Salary
0       39          State-gov  ...   United-States   <=50K
1       50   Self-emp-not-inc  ...   United-States   <=50K
2       38            Private  ...   United-States   <=50K
3       53            Private  ...   United-States   <=50K
4       28            Private  ...            Cuba   <=50K
   ...                ...  ...             ...     ...
30156   27            Private  ...   United-States   <=50K
30157   40            Private  ...   United-States    >50K
30158   58            Private  ...   United-States   <=50K
30159   22            Private  ...   United-States   <=50K
30160   52       Self-emp-inc  ...   United-States    >50K

[30161 rows x 14 columns]

'''
data.head()
data1.head()

data.tail()
data1.tail()

data.shape
#(15060, 14)

data1.shape
#(30161, 14)

data.describe()

'''
   age   educationno   capitalgain   capitalloss  hoursperweek
count  15060.000000  15060.000000  15060.000000  15060.000000  15060.000000
mean      38.768327     10.112749   1120.301594     89.041899     40.951594
std       13.380676      2.558727   7703.181842    406.283245     12.062831
min       17.000000      1.000000      0.000000      0.000000      1.000000
25%       28.000000      9.000000      0.000000      0.000000     40.000000
50%       37.000000     10.000000      0.000000      0.000000     40.000000
75%       48.000000     13.000000      0.000000      0.000000     45.000000
max       90.000000     16.000000  99999.000000   3770.000000     99.000000

'''
data1.describe()
'''
    age   educationno   capitalgain   capitalloss  hoursperweek
count  30161.000000  30161.000000  30161.000000  30161.000000  30161.000000
mean      38.438115     10.121316   1092.044064     88.302311     40.931269
std       13.134830      2.550037   7406.466611    404.121321     11.980182
min       17.000000      1.000000      0.000000      0.000000      1.000000
25%       28.000000      9.000000      0.000000      0.000000     40.000000
50%       37.000000     10.000000      0.000000      0.000000     40.000000
75%       47.000000     13.000000      0.000000      0.000000     45.000000
max       90.000000     16.000000  99999.000000   4356.000000     99.000000

'''
data.info()
data1.info()

data.isnull().sum()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15060 entries, 0 to 15059
Data columns (total 14 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            15060 non-null  int64 
 1   workclass      15060 non-null  object
 2   education      15060 non-null  object
 3   educationno    15060 non-null  int64 
 4   maritalstatus  15060 non-null  object
 5   occupation     15060 non-null  object
 6   relationship   15060 non-null  object
 7   race           15060 non-null  object
 8   sex            15060 non-null  object
 9   capitalgain    15060 non-null  int64 
 10  capitalloss    15060 non-null  int64 
 11  hoursperweek   15060 non-null  int64 
 12  native         15060 non-null  object
 13  Salary         15060 non-null  object
dtypes: int64(5), object(9)
memory usage: 1.6+ MB

'''
data1.isnull().sum()
#there is no null values

sns.boxplot(data["age"])

num= data.select_dtypes(include=["int64","float64"]).columns
for col in num:
    sns.boxplot(data[col])
    plt.title(f"boxplot for {col}")
    plt.show()

sns.distplot(data["age"])

#drop the columns
data=data.drop(["capitalgain"],axis=1)
data=data.drop(["capitalloss"],axis=1)
data1=data1.drop(["capitalgain"],axis=1)
data1=data1.drop(["capitalloss"],axis=1)

#remove the outliers
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="age",tail="both")
data["age"]=winsor.fit_transform(data[["age"]])
sns.boxplot(data["age"]) 

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="educationno",tail="both")
data["educationno"]=winsor.fit_transform(data[["educationno"]])
sns.boxplot(data["educationno"])

winsor=Winsorizer(capping_method="iqr",fold=1.5,variables="hoursperweek",tail="both")
data["hoursperweek"]=winsor.fit_transform(data[["hoursperweek"]])
sns.boxplot(data["hoursperweek"])

from sklearn.preprocessing import LabelEncoder

# Identify categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data.workclass=labelencoder.fit_transform(data.workclass)
data.education=labelencoder.fit_transform(data.education)
data.maritalstatus=labelencoder.fit_transform(data.maritalstatus)
data.occupation=labelencoder.fit_transform(data.occupation)
data.relationship=labelencoder.fit_transform(data.relationship)
data.race=labelencoder.fit_transform(data.race)
data.sex=labelencoder.fit_transform(data.sex )
data.native=labelencoder.fit_transform(data.native)
data.Salary=labelencoder.fit_transform(data.Salary)

data1.workclass=labelencoder.fit_transform(data1.workclass)
data1.education=labelencoder.fit_transform(data1.education)
data1.maritalstatus=labelencoder.fit_transform(data1.maritalstatus)
data1.occupation=labelencoder.fit_transform(data1.occupation)
data1.relationship=labelencoder.fit_transform(data1.relationship)
data1.race=labelencoder.fit_transform(data1.race)
data1.sex=labelencoder.fit_transform(data1.sex )
data1.native=labelencoder.fit_transform(data1.native)
data1.Salary=labelencoder.fit_transform(data1.Salary)
#

# Encode categorical variables
data_encoded = data.copy()
for col in categorical_cols:
    data_encoded[col] = label_encoder.fit_transform(data[col])

# Create correlation matrix and heatmap
plt.figure(figsize=(20,20))
tc = data_encoded.corr()
sns.heatmap(tc, cmap="YlGnBu", annot=True, fmt=".2f")
plt.show()


from sklearn.svm import SVC
train_X=data.iloc[:,:10]
train_y=data.iloc[:,10]
test_X=data1.iloc[:,:10]
test_y=data1.iloc[:,10]
#Kernel linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_X,train_y)
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)
#RBF
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)
#0.9119061039090216