from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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

# rbf SVM
model1 = svm.SVC(kernel='rbf', probability=True)
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
prob = model1.predict_proba(x_test)

# feature importance
per_importance = permutation_importance(model1, x_test, y_test)
feature_names = x_train.columns
features = np.array(feature_names)
sorted_idx = per_importance.importances_mean.argsort()
print(features[sorted_idx])
print(per_importance.importances_mean[sorted_idx])

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred
df['probability0'] = prob[:, 0]
df['probability1'] = prob[:, 1]
df1 = pd.concat([df["rank"], df["true"], df["predicted"], df["probability0"], df["probability1"]], axis=1)
final_df = df1.sort_values(by=['probability1'], ascending=False)
print(final_df.head(16))

# confusion matrix
svm_cm = confusion_matrix(y_pred, y_test, labels=model1.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('SVM_images/rbfSVM_Classifier')
plt.clf()

# ROC curve
svc_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % svc_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('SVM_images/rbfSVM_ROC')
plt.close()

# hyperparameter tuning
# the best parameters are C = 10 and gamma = 0.1
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, scoring='recall', refit=True, verbose=3)
#grid.fit(x_train, y_train)
#print(grid.best_params_)

# rbf SVM with hyperparameter tuning
model2 = svm.SVC(C=10, gamma=0.1, kernel='rbf', probability=True)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
prob = model2.predict_proba(x_test)

# feature importance
per_importance = permutation_importance(model2, x_test, y_test)
feature_names = x_train.columns
features = np.array(feature_names)
sorted_idx = per_importance.importances_mean.argsort()
print(features[sorted_idx])
print(per_importance.importances_mean[sorted_idx])

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred
df['probability0'] = prob[:, 0]
df['probability1'] = prob[:, 1]
df1 = pd.concat([df["rank"], df["true"], df["predicted"], df["probability0"], df["probability1"]], axis=1)
final_df = df1.sort_values(by=['probability1'], ascending=False)
print(final_df.head(16))

# final confusion matrix
svm_cm = confusion_matrix(y_pred, y_test, labels=model2.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('SVM_images/final_rbfSVM_Classifier')
plt.clf()

# final ROC curve
svc_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % svc_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('SVM_images/final_rbfSVM_ROC')
plt.close()
