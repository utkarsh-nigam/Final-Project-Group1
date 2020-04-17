import pandas as pd
data=pd.read_csv("HR-Employee-Attrition.csv")
print(data.head())
print(data.columns)
print(data.dtypes)

categorical_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']
data[categorical_cols] = data[categorical_cols].astype('category')
data['EmployeeNumber']=data['EmployeeNumber'].astype(str)
print(data.dtypes)
