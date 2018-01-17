from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

iris_dataset = load_iris()

print(iris_dataset.keys())

# print(iris_dataset['DESCR'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: %s" % str(X_train.shape))
print("y_train shape: %s" % str(y_train.shape))

print("X_test shape: %s" % str(X_test.shape))
print("y_test shape: %s" % str(y_test.shape))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                  marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.savefig(r"figure_1.png")

print(type(grr))

print("done")
