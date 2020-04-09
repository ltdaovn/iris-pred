# Nạp các gói thư viện cần thiết
import pandas as pd
from sklearn import tree


# Đọc dữ liệu iris từ UCI (https://archive.ics.uci.edu/ml/datasets/Iris)
# hoặc từ thư viện scikit-learn
# Tham khảo https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
#print(iris)
columns=["Petal length","Petal Width","Sepal Length","Sepal Width"];
X = pd.DataFrame(iris.data, columns=columns)
y = iris.target
#print(X.head())
#print(y)


# Xây dựng mô hình với giải thuật Cây quyết định
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X, y)

from joblib import dump, load
dump(model, 'Iris.DecisionTree.joblib') 
