import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


df = pd.read_csv('co2.csv')

print(df.head(10))
print(df.describe())

sns.countplot(x='out1', data=df)

plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True)


x = df.drop("out1", axis=1)
x = x.drop("fuelcomb", axis=1)
x = x.drop("cylandr", axis=1)
y = df.out1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg_linear= linear_model.LinearRegression()

reg_linear.fit(X_train, y_train)

y_test_pred = reg_linear.predict(X_test)

print(X_test.size)

test=np.array([[3]])

khoroji = reg_linear.predict(test)

print(khoroji)


plt.show()
