import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def customScorer(y_true, y_pred):
    count = 1
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred == 1:
            count += 1
    return count / 16


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    X_train = train.drop(['target', 'Unnamed: 0'], axis=1)
    y_train = train['target']
    X_test = test.drop(['target', 'Unnamed: 0'], axis=1)
    y_test = test['target']

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'n_estimators': [50, 100, 200],
        'random_state': [20]
    }

    # define the Gradient Boosting model with default hyperparameters
    model = GradientBoostingClassifier(random_state=42)
    print(' ' * 9 + "Original Model")
    print('*' * 30)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    pos_probs = proba[:, 1]
    idx = np.argsort(-pos_probs)[:16]
    y_pred = [1 if i in idx else 0 for i in range(len(y_test))]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 1:
            count += 1

    print('Metrics:')
    print('# of top 16 Players Correct:', count)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

    print()
    print(' ' * 8 + "Fine Tuned Model")
    print('*' * 30)
    # define the grid search
    model = GradientBoostingClassifier()

    grid_search = GridSearchCV(model, param_grid, scoring=customScorer, cv=5)

    # fit the grid search on the training data
    grid_search.fit(X_train, y_train)

    print('Best parameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    best_model = grid_search.best_estimator_

    proba = best_model.predict_proba(X_test)
    pos_probs = proba[:, 1]
    idx = np.argsort(-pos_probs)[:16]
    y_pred = [1 if i in idx else 0 for i in range(len(y_test))]

    importances = best_model.feature_importances_
    # Sort the features by importance score in descending order
    sorted_idx = importances.argsort()[::-1]

    # Print the top 5 features and their importance scores
    print()
    print("Top 5 features:")
    for idx in sorted_idx[:5]:
        print(f"{X_train.columns[idx]}: {importances[idx]}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 1:
            count += 1

    print()
    print('Metrics:')
    print('# of top 16 Players Correct:', count)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)


if __name__ == '__main__':
    main()
