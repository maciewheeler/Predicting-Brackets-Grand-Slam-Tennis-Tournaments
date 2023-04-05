import itertools

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score

import pandas as pd
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

# feature importance/selection (dimensionality reduction)
# ['A', 'B', 'E', 'F', 'G', 'H', 'I', 'K', 'M', 'N', 'O', 'P']
x_train_new = x_train.copy()
x_test_new = x_test.copy()

x_train_new.rename(columns={'1st_serve':"A", '1st_serve_points_won':"B", '2nd_serve_points_won': "C",
                            'break_points_saved': "D", 'service_games_won': "E", '1st_serve_return_points_won': "F",
                            '2nd_serve_return_points_won': "G", 'break_points_converted': "H", 'return_games_won': "I",
                            'rank': "J", 'height (cm)': "K", 'matches_played': "L", 'aces_per_match': "M",
                            'double_faults_per_match': "N", 'break_points_opportunities_per_match': "O",
                            'break_points_faced_per_match': "P"}, inplace=True)

x_test_new.rename(columns={'1st_serve':"A", '1st_serve_points_won':"B", '2nd_serve_points_won': "C",
                            'break_points_saved': "D", 'service_games_won': "E", '1st_serve_return_points_won': "F",
                            '2nd_serve_return_points_won': "G", 'break_points_converted': "H", 'return_games_won': "I",
                            'rank': "J", 'height (cm)': "K", 'matches_played': "L", 'aces_per_match': "M",
                            'double_faults_per_match': "N", 'break_points_opportunities_per_match': "O",
                            'break_points_faced_per_match': "P"}, inplace=True)

for i in range(1, 17):
    c = itertools.combinations("ABCDEFGHIJKLMNOP", i)
    for item in c:
        ls = list(item)
        model = svm.SVC(kernel='linear')
        model.fit(x_train_new[ls], y_train)
        y_pred = model.predict(x_test_new[ls])
        accuracy = recall_score(y_train, y_pred)

        if accuracy > 0.9:
            print(ls, "  Accuracy on test set:{:.4%}".format(accuracy))

# x_train = pd.concat([x_train["1st_serve"], x_train["1st_serve_points_won"], x_train["service_games_won"],
#                      x_train["1st_serve_return_points_won"], x_train["2nd_serve_return_points_won"],
#                      x_train["break_points_converted"], x_train["return_games_won"], x_train["height (cm)"],
#                      x_train["aces_per_match"], x_train["double_faults_per_match"],
#                      x_train["break_points_opportunities_per_match"], x_train["break_points_faced_per_match"]], axis=1)
#
# x_test = pd.concat([x_test["1st_serve"], x_test["1st_serve_points_won"], x_test["service_games_won"],
#                      x_test["1st_serve_return_points_won"], x_test["2nd_serve_return_points_won"],
#                      x_test["break_points_converted"], x_test["return_games_won"], x_test["height (cm)"],
#                      x_test["aces_per_match"], x_test["double_faults_per_match"],
#                      x_test["break_points_opportunities_per_match"], x_test["break_points_faced_per_match"]], axis=1)

# linear SVM
model1 = svm.SVC(kernel='linear', probability=True)
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
prob = model1.predict_proba(x_test)

# feature importance
print(model1.coef_)
print(model1.feature_names_in_)

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred
df['probability0'] = prob[:, 0]
df['probability1'] = prob[:, 1]
df1 = pd.concat([df["true"], df["predicted"], df["probability0"], df["probability1"]], axis=1)
final_df = df1.sort_values(by=['probability1'], ascending=False)
print(final_df.head(16))

# confusion matrix
svm_cm = confusion_matrix(y_pred, y_test, labels=model1.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('SVM_images/linearSVM_Classifier')
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
plt.savefig('SVM_images/linearSVM_ROC')
plt.close()

# hyperparameter tuning
# the best parameter is C = 100
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, scoring='recall', refit=True, verbose=3)
#grid.fit(x_train, y_train)
#print(grid.best_params_)

# linear SVM with hyperparameter tuning
model2 = svm.SVC(C=100, kernel='linear', probability=True)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
prob = model2.predict_proba(x_test)

# feature importance
print(model2.coef_)
print(model2.feature_names_in_)

# result dataframe
df = x_test.copy()
df['true'] = y_test
df['predicted'] = y_pred
df['probability0'] = prob[:, 0]
df['probability1'] = prob[:, 1]
df1 = pd.concat([df["true"], df["predicted"], df["probability0"], df["probability1"]], axis=1)
final_df = df1.sort_values(by=['probability1'], ascending=False)
print(final_df.head(16))

# final confusion matrix
svm_cm = confusion_matrix(y_pred, y_test, labels=model2.classes_)
sns.heatmap(svm_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('SVM Classifier')
plt.savefig('SVM_images/final_linearSVM_Classifier')
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
plt.savefig('SVM_images/final_linearSVM_ROC')
plt.close()
