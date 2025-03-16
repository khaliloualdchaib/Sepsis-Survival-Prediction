import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import numpy as np

def show_tree(dt):
    plt.figure(figsize=(24,16))
    tree.plot_tree(dt)
    plt.savefig('fully_grown_decision_tree.png',facecolor='white',bbox_inches="tight")

def evaluateModel(dt, X, y):
    y_pred = dt.predict(X)
    # print metrics
    print("Precision = " + str(np.round(precision_score(y, y_pred),3)))
    print("Recall = " + str(np.round(recall_score(y, y_pred),3)))
    print("F1 = " + str(np.round(f1_score(y, y_pred),3)))