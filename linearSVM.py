from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train = train.drop(columns=['Unnamed: 0'])
test = test.drop(columns=['Unnamed: 0'])

# split data
y_train = train["target"]
x_train = train.drop(columns=["target"])
y_test = test["target"]
x_test = test.drop(columns=["target"])

# scale data
sc = StandardScaler()
sc.fit(x_train)
x_train = pd.DataFrame(sc.transform(x_train), index=x_train.index, columns=x_train.columns)
x_test = pd.DataFrame(sc.transform(x_test), index=x_test.index, columns=x_test.columns)

# 1st round rbf SVM
model1 = svm.SVC(kernel='linear')
model1.fit(x_train, y_train)
y_pred = model1.predict(x_train)
y_pred1 = model1.predict(x_test)
print("Train accuracy:", metrics.accuracy_score(y_train, y_pred))
print("Test accuracy:", metrics.accuracy_score(y_test, y_pred1))
print("Precision:", metrics.precision_score(y_test, y_pred1))
print("Recall:", metrics.recall_score(y_test, y_pred1))

# feature importance/selection (dimensionality reduction)
# 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P' features give greatest accuracy and keeps the most possible features
# x_train_new = x_train.copy()
# x_test_new = x_test.copy()
#
# x_train_new.rename(columns={'1st_serve':"A", '1st_serve_points_won':"B", '2nd_serve_points_won': "C",
#                             'break_points_saved': "D", 'service_games_won': "E", '1st_serve_return_points_won': "F",
#                             '2nd_serve_return_points_won': "G", 'break_points_converted': "H", 'return_games_won': "I",
#                             'rank': "J", 'height (cm)': "K", 'matches_played': "L", 'aces_per_match': "M",
#                             'double_faults_per_match': "N", 'break_points_opportunities_per_match': "O",
#                             'break_points_faced_per_match': "P"}, inplace=True)
#
# x_test_new.rename(columns={'1st_serve':"A", '1st_serve_points_won':"B", '2nd_serve_points_won': "C",
#                             'break_points_saved': "D", 'service_games_won': "E", '1st_serve_return_points_won': "F",
#                             '2nd_serve_return_points_won': "G", 'break_points_converted': "H", 'return_games_won': "I",
#                             'rank': "J", 'height (cm)': "K", 'matches_played': "L", 'aces_per_match': "M",
#                             'double_faults_per_match': "N", 'break_points_opportunities_per_match': "O",
#                             'break_points_faced_per_match': "P"}, inplace=True)
#
# for i in range(1, 17):
#     c = itertools.combinations("ABCDEFGHIJKLMNOP", i)
#     for item in c:
#         ls = list(item)
#         model = svm.SVC(kernel='linear')
#         model.fit(x_train_new[ls], y_train)
#         y_pred = model.predict(x_train_new[ls])
#         y_pred1 = model.predict(x_test_new[ls])
#         accuracy = metrics.accuracy_score(y_train, y_pred)
#         accuracy1 = metrics.accuracy_score(y_test, y_pred1)
#
#         if accuracy1 > 0.9:
#             print(ls, "  Accuracy on training set:{:.4%}".format(accuracy),
#             " Accuracy on test set:{:.4%}".format(accuracy1))

x_train = pd.concat([x_train["1st_serve_points_won"], x_train["2nd_serve_points_won"],
                     x_train["1st_serve_return_points_won"], x_train["2nd_serve_return_points_won"],
                             x_train["break_points_converted"], x_train["return_games_won"], x_train["rank"],
                     x_train["height (cm)"], x_train["matches_played"], x_train["aces_per_match"],
                     x_train["double_faults_per_match"], x_train["break_points_opportunities_per_match"],
                             x_train["break_points_faced_per_match"]], axis=1)

x_test = pd.concat([x_test["1st_serve_points_won"], x_test["2nd_serve_points_won"],
                     x_test["1st_serve_return_points_won"], x_test["2nd_serve_return_points_won"],
                             x_test["break_points_converted"], x_test["return_games_won"], x_test["rank"],
                     x_test["height (cm)"], x_test["matches_played"], x_test["aces_per_match"],
                     x_test["double_faults_per_match"], x_test["break_points_opportunities_per_match"],
                             x_test["break_points_faced_per_match"]], axis=1)

# SVM with dimensionality reduction
model2 = svm.SVC(kernel='linear', probability=True)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_train)
y_pred1 = model2.predict(x_test)
prob = model2.predict_proba(x_test)
print("Train accuracy:", metrics.accuracy_score(y_train, y_pred))
print("Test accuracy:", metrics.accuracy_score(y_test, y_pred1))
print("Precision:", metrics.precision_score(y_test, y_pred1))
print("Recall:", metrics.recall_score(y_test, y_pred1))

# feature importance
print(model2.coef_)
print(model2.feature_names_in_)

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred1
df['probability0'] = prob[:, 0]
df['probability1'] = prob[:, 1]
df1 = pd.concat([df["rank"], df["true"], df["predicted"], df["probability0"], df["probability1"]], axis=1)
final_df = df1.sort_values(by=['probability1'], ascending=False)
print(final_df.head(16))

# confusion matrix
svm_cm = confusion_matrix(y_pred1, y_test, labels=model2.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('linearSVM_Classifier')
plt.clf()

# ROC curve
svc_roc_auc = roc_auc_score(y_test, y_pred1)
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % svc_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('linearSVM_ROC')
plt.close()

# hyperparameter tuning
# the best parameters are C = 0.1
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, refit=True, verbose=3)
#grid.fit(x_train, y_train)
#print(grid.best_params_)

# SVM with hyperparameters and dimensionality reduction
model3 = svm.SVC(C=0.01, kernel='linear', probability=True)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_train)
y_pred1 = model3.predict(x_test)
prob = model3.predict_proba(x_test)
print("Accuracy:", metrics.accuracy_score(y_train, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred1))
print("Precision:", metrics.precision_score(y_test, y_pred1))
print("Recall:", metrics.recall_score(y_test, y_pred1))

# final confusion matrix
svm_cm = confusion_matrix(y_pred1, y_test, labels=model3.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('final_linearSVM_Classifier')
plt.clf()

# final ROC curve
svc_roc_auc = roc_auc_score(y_test, y_pred1)
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % svc_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('final_linearSVM_ROC')
plt.close()
