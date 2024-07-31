import os
import pickle

import graphviz
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

file_path = r"data\raw\hand.csv"

# ファイルの読み込み
df = pd.read_csv(file_path, header=0)
# print(df.shape)  # (1221, 64)
# print(df.head())

# データの描画
# print(df.iloc[:, 0])
# print(df.iloc[:, 1])
# print(df.iloc[:, 2])
# plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c=df.iloc[:, 0])
# 軸名
# plt.xlabel('WRIST_x')
# plt.ylabel('WRIST_y')
# plt.show()

# データの分割
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.shape)  # (976, 63)
# print(X_test.shape)  # (245, 63)
# print(Y_train.shape)  # (976,)
# print(Y_test.shape)  # (245,)

# 決定木
clf_model = DecisionTreeClassifier(max_depth=3)
clf_model.fit(X_train, Y_train)
print(f"train score: {clf_model.score(X_train, Y_train)}")
print(f"test score: {clf_model.score(X_test, Y_test)}")

# 決定木の可視化
dot_data = export_graphviz(clf_model)
graph = graphviz.Source(dot_data)
# graph.view()


os.makedirs(r"models", exist_ok=True)
with open(r"models\clf_model.pickle", mode="wb") as f:
    pickle.dump(clf_model, f)
