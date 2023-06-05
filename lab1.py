import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('train.csv', sep=',', index_col=False, dtype='unicode')
pd.set_option('display.max_columns', None)

plt.figure(figsize=(5, 5))
sns.countplot(x='Credit_Score', data=data)
plt.title('Целевой класс')
plt.show()


cols = ['Credit_History_Age', 'Delay_from_due_date', 'Interest_Rate']
for col in cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=col)
    plt.title(f'{col}')
    plt.show()


X = data.drop('Credit_Score',axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X.head()

X = data.drop("Credit_Score",axis=1)
numeric = X.select_dtypes(exclude="object").columns

low_cardinality = [col for col in numeric if data[col].nunique() <= 30]
high_cardinality = [col for col in numeric if data[col].nunique() > 30]

plt.figure(figsize=(8, 6))
arr = np.ones_like(data[high_cardinality].corr())
mask = np.triu(arr)
sns.heatmap(data[high_cardinality].corr(), cbar=False, annot=True, fmt=".2g", mask=mask)
