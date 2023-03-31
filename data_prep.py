import pandas as pd

train = pd.read_csv('data/aus-open-player-stats-2017.csv')
test = pd.read_csv('data/aus-open-player-stats-2018.csv')
results2018 = pd.read_csv('data/aus-open-2018.csv')
results2019 = pd.read_csv('data/aus-open-2019.csv')

top16_2018 = list(results2018[results2018["Round"] == "3rd Round"]["WRank"])
top16_2019 = list(results2019[results2019["Round"] == "3rd Round"]["WRank"])

columns_to_remove = ["name", "year", "service_games_played", "return_games_played", "total_service_points_won",
                     "return_points_won", "total_points_won"]

train = train.drop(columns=columns_to_remove)
test = test.drop(columns=columns_to_remove)

columns_to_decimal = ["1st_serve", "1st_serve_points_won", "2nd_serve_points_won", "break_points_saved",
                     "service_games_won", "1st_serve_return_points_won", "2nd_serve_return_points_won",
                      "break_points_converted", "return_games_won"]

for col in columns_to_decimal:
    count1 = 0
    count2 = 0

    for i in train[col]:
        percent = float(i.strip("%")) / 100
        train.loc[count1, col] = percent
        count1 = count1 + 1
    for j in test[col]:
        percent = float(j.strip("%")) / 100
        test.loc[count2, col] = percent
        count2 = count2 + 1

train = train[~(train['aces'] == 0)]
test = test[~(test['aces'] == 0)]

train['target'] = train['rank'].apply(lambda x: 1 if x in top16_2018 else 0)
test['target'] = test['rank'].apply(lambda x: 1 if x in top16_2019 else 0)

derived_attributes = ["aces", "double_faults", "break_points_opportunities", "break_points_faced"]

for attribute in derived_attributes:
    derived_attribute = attribute + "_per_match"
    train[derived_attribute] = train[attribute] / train["matches_played"]
    train = train.drop(columns=attribute)

    test[derived_attribute] = test[attribute] / test["matches_played"]
    test = test.drop(columns=attribute)

train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
