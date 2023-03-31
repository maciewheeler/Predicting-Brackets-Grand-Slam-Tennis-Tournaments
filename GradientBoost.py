import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV




def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test = test.drop('target', axis=1)
    y_test = test['target']

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [50, 100, 200]
    }

    # define the Gradient Boosting model with default hyperparameters
    model = GradientBoostingClassifier()

    # define the grid search
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)

    # fit the grid search on the training data
    # model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)

    print('Best parameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    best_model = grid_search.best_estimator_

    proba = best_model.predict_proba(X_test)
    pos_probs = proba[:, 1]
    idx = np.argsort(-pos_probs)[:16]
    y_pred = [1 if i in idx else 0 for i in range(len(y_test))]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 1:
            count += 1
    print('Count:', count)
    print('Accuracy:', accuracy)
    print('F1:', f1)

if __name__ == '__main__':
    main()
