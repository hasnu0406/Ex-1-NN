## NAME: HASNA MUBARAK AZEEM
## REGISTER NO.: 212223240052
## EX. NO.1
## DATE: 19/08/25
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv")
print(df)
X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,-1].values
print(y)
print(df.isnull().sum())
print(df.duplicated())
print(df['Age'].describe)
df=df.drop(['Surname','Geography','Gender'],axis=1)
df.head()
scaler=MinMaxScaler()
new_df=pd.DataFrame(scaler.fit_transform(df))
print(new_df)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```


## OUTPUT:
<img width="580" height="649" alt="image" src="https://github.com/user-attachments/assets/9de2176c-7efb-47c0-b76d-92d5562bb5f0" />
<img width="504" height="157" alt="image" src="https://github.com/user-attachments/assets/327de2a2-5901-4eaa-9a6d-fc0e8bc3c138" />
<img width="193" height="57" alt="image" src="https://github.com/user-attachments/assets/d62c7d7f-1e09-4dbc-8742-a26b00dee025" />
<img width="479" height="271" alt="image" src="https://github.com/user-attachments/assets/86548720-de4e-4f13-b3f9-127e2378c3e9" />
<img width="1078" height="207" alt="image" src="https://github.com/user-attachments/assets/77261092-992a-4208-92cc-e815a17f770e" />
<img width="784" height="576" alt="image" src="https://github.com/user-attachments/assets/a0847014-ce62-4a1a-9238-1918c171e063" />
<img width="473" height="348" alt="image" src="https://github.com/user-attachments/assets/2cc86310-b6ea-40a6-b18a-027b94ebe3c4" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


