import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

df=pd.read_csv('https://raw.githubusercontent.com/safal/DS-ML/refs/heads/main/Iris.csv')

print(df.head())
print(df.shape)
x=df.drop(['Species','Id'],axis=1)
y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
model=DecisionTreeClassifier(criterion='entropy',min_samples_split=50)
model.fit(x_train,y_train)
y_test_predict=model.predict(x_test)
print(accuracy_score(y_test,y_test_predict))
tree.plot_tree(model)
plt.show()