import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

df = pd.read_csv("./customer_churn_dataset-training-master.csv")

print("Reading data sucessfully")

print(df.shape)

df=df.dropna()

df=df.drop(columns=['CustomerID','Subscription Type'],axis=1)

x = df.drop('Churn',axis=1)
y=df['Churn']

x_num = x.select_dtypes(exclude='object')
x_cat = x.select_dtypes(include='object')

preprocessor = ColumnTransformer(
    transformers=[
        ('num',Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ]),x_num.columns),
        ('cat',Pipeline(steps=[
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ]),x_cat.columns)
    ]
)

pipeline=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


pipeline.fit(x_train,y_train)

y_pred=pipeline.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred))

print(metrics.confusion_matrix(y_test,y_pred))