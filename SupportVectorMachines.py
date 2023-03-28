from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# get data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# split data
y_train = train["target"]
x_train = train.drop(columns=["target"])
y_test = test["target"]
x_test = test.drop(columns=["target"])

# linear kernel
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
prob = clf.predict_proba(x_test)[:, 1]
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred
df['probability'] = prob
final_df = df.sort_values(by=['probability'], ascending=False)
print(final_df.head(16))

# confusion matrix
svm_cm = confusion_matrix(y_pred, y_test, labels=clf.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('SVM_Classifier')
plt.clf()

# precision, recall, F1, support
print(classification_report(y_test, y_pred))

# ROC curve
svc_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, prob)
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % svc_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('SVM_ROC')
plt.close()

# hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train)
print(grid.best_params_)

# rbf kernel
clf = svm.SVC(C=0.01, gamma=0.01, kernel='rbf', probability=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
prob = clf.predict_proba(x_test)[:, 1]
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
