import pandas as pd

train = pd.read_csv('data/aus-open-player-stats-2018.csv')
test = pd.read_csv('data/aus-open-2019.csv')

columns_to_remove = ["name", "year", "service_games_played", "return_games_played", "total_service_points_won",
                     "return_points_won", "total_points_won"]

train = train.drop(columns=columns_to_remove)

columns_to_decimal = ["1st_serve", "1st_serve_points_won", "2nd_serve_points_won", "break_points_saved",
                     "service_games_won", "1st_serve_return_points_won", "2nd_serve_return_points_won",
                      "break_points_converted", "return_games_won"]

for col in columns_to_decimal:
    count = 0
    for i in train[col]:
        percent = float(i.strip("%")) / 100
        train.loc[count, col] = percent
        count = count + 1

derived_attributes = ["aces", "double_faults", "break_points_opportunities", "break_points_faced"]

for attribute in derived_attributes:
    derived_attribute = attribute + "_per_match"
    train[derived_attribute] = train[attribute] / train["matches_played"]
    train = train.drop(columns=attribute)

train.to_csv('data/train.csv')

new16 = test[test["Round"] == "3rd Round"]
new_test16 = new16["Winner"]

new_test16.to_csv("data/test.csv")
