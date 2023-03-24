import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    data = pd.read_csv('data/train.csv')
    data2 = pd.read_csv('data/aus-open-player-stats-2018.csv')
    test = pd.read_csv('data/test.csv')
    X = data.drop('rank', axis=1)
    y = data['rank']
    players = data2['name']
    winners = set(test['Winner'].apply(lambda x: x))
    def label(x):
        name = x.split(' ')
        first_name = name[0]
        last_name = name[1]
        if
    data['winner'] = data2['name'].apply(lambda x: 1 if x.split(' ')[1] in winners else 0)
    print(data['winner'].sum())


    # print()

    # split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # define the Gradient Boosting model with default hyperparameters
    # model = GradientBoostingClassifier()
    #
    # # fit the model on the training data
    # model.fit(X_train, y_train)
    #
    # # make predictions on the testing data
    #
    # # use the trained model to predict the top 16 players
    # top_player_indices = model.predict(X_test).argsort()[:16]
    #
    # top_players = [players[i] for i in top_player_indices]
    # top_ranks = [y[i] for i in top_player_indices]
    # count = 0
    # # print the top 16 players and their predicted rankings
    # for i, (player, rank) in enumerate(zip(top_players, top_ranks), 1):
    #     last_name = player.split(' ')[1]
    #     if last_name in winners:
    #         count += 1
    # print(count)

if __name__ == '__main__':
    main()
