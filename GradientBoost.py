import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # data2 = pd.read_csv('data/aus-open-player-stats-2018.csv')
    # test = pd.read_csv('data/test.csv')
    # players = data2['name']
    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test = test.drop('target', axis=1)
    y_test = test['target']


    # split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define the Gradient Boosting model with default hyperparameters
    model = GradientBoostingClassifier()

    # fit the model on the training data
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    pos_probs = proba[:, 1]
    idx = np.argsort(-pos_probs)[:16]
    y_pred = [1 if i in idx else 0 for i in range(len(y_test))]
    # make predictions on the testing data
    # y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 1:
            count += 1
    print('Count:', count)
    print('Accuracy:', accuracy)
    print('F1:', f1)

    # use the trained model to predict the top 16 players
    # top_player_indices = model.predict(X_test)
    # print(top_player_indices)
    # print(sum(top_player_indices))

    # top_players = [players[i] for i in top_player_indices]
    # top_ranks = [y[i] for i in top_player_indices]
    # count = 0
    # print the top 16 players and their predicted rankings
    # for i, (player, rank) in enumerate(zip(top_players, top_ranks), 1):
    #     last_name = player.split(' ')[1]
    #     if last_name in winners:
    #         count += 1
    # print(count)

if __name__ == '__main__':
    main()
