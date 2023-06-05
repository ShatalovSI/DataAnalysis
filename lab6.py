import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree  import DecisionTreeClassifier as dtClassifier

pd.set_option('display.max.columns', None)

df = pd.read_csv('train.csv', sep=',', encoding='utf-8')
df_test = pd.read_csv('test.csv', sep=',', encoding='utf-8')

mdf = df[['Credit_Score','Changed_Credit_Limit', 'Payment_of_Min_Amount', 'Credit_Mix', 'Delay_from_due_date', 'Annual_Income', 'Monthly_Inhand_Salary', 'Age', 'Monthly_Balance', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 'Payment_Behaviour', 'Credit_History_Age', 'Num_Bank_Accounts', 'Credit_Utilization_Ratio']]
x = mdf.drop(['Credit_Score'] , axis = 1).values
y = mdf['Credit_Score' ].values
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size= 0.25 , random_state=42)
print([x_train.shape, y_train.shape])

dt = dtClassifier(max_features=14, max_depth=8)
dt.fit(x_train , y_train)
dt_score = dt.score(x_train , y_train)
dt_score_t = dt.score(x_test , y_test)

y_pred3 = dt.predict(x_test)
dd = pd.DataFrame({"Y_test" : y_test , "y_pred3": y_pred3})
plt.figure(figsize=(10,8))
plt.plot(dd[:100])
plt.legend(["Actual" , "Predicted"])