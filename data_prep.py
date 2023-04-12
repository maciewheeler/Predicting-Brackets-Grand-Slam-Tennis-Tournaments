import pandas as pd

train16 = pd.read_csv('data/aus-open-player-stats-2017.csv')
train32 = pd.read_csv('data/aus-open-player-stats-2017.csv')
test16 = pd.read_csv('data/aus-open-player-stats-2018.csv')
test32 = pd.read_csv('data/aus-open-player-stats-2018.csv')
results2018 = pd.read_csv('data/aus-open-2018.csv')
results2019 = pd.read_csv('data/aus-open-2019.csv')

top16_2018 = list(results2018[results2018["Round"] == "3rd Round"]["WRank"])
top32_2018 = list(results2018[results2018["Round"] == "2nd Round"]["WRank"])
top16_2019 = list(results2019[results2019["Round"] == "3rd Round"]["WRank"])
top32_2019 = list(results2019[results2019["Round"] == "2nd Round"]["WRank"])

train16['target'] = train16['rank'].apply(lambda x: 1 if x in top16_2018 else 0)
train32['target'] = train32['rank'].apply(lambda x: 1 if x in top32_2018 else 0)
test16['target'] = test16['rank'].apply(lambda x: 1 if x in top16_2019 else 0)
test32['target'] = test32['rank'].apply(lambda x: 1 if x in top32_2019 else 0)

columns_to_remove = ["name", "year", "service_games_played", "return_games_played", "total_service_points_won",
                     "return_points_won", "total_points_won"]

train16 = train16.drop(columns=columns_to_remove)
train32 = train32.drop(columns=columns_to_remove)
test16 = test16.drop(columns=columns_to_remove)
test32 = test32.drop(columns=columns_to_remove)

columns_to_decimal = ["1st_serve", "1st_serve_points_won", "2nd_serve_points_won", "break_points_saved",
                     "service_games_won", "1st_serve_return_points_won", "2nd_serve_return_points_won",
                      "break_points_converted", "return_games_won"]

for col in columns_to_decimal:
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0

    for i in train16[col]:
        percent = float(i.strip("%")) / 100
        train16.loc[count1, col] = percent
        count1 = count1 + 1
    for m in train32[col]:
        percent = float(m.strip("%")) / 100
        train32.loc[count2, col] = percent
        count2 = count2 + 1
    for j in test16[col]:
        percent = float(j.strip("%")) / 100
        test16.loc[count3, col] = percent
        count3 = count3 + 1
    for n in test32[col]:
        percent = float(n.strip("%")) / 100
        test32.loc[count4, col] = percent
        count4 = count4 + 1

train16 = train16[~(train16['aces'] == 0)]
train32 = train32[~(train32['aces'] == 0)]
test16 = test16[~(test16['aces'] == 0)]
test32 = test32[~(test32['aces'] == 0)]

derived_attributes = ["aces", "double_faults", "break_points_opportunities", "break_points_faced"]

for attribute in derived_attributes:
    derived_attribute = attribute + "_per_match"
    train16[derived_attribute] = train16[attribute] / train16["matches_played"]
    train16 = train16.drop(columns=attribute)

    train32[derived_attribute] = train32[attribute] / train32["matches_played"]
    train32 = train32.drop(columns=attribute)

    test16[derived_attribute] = test16[attribute] / test16["matches_played"]
    test16 = test16.drop(columns=attribute)

    test32[derived_attribute] = test32[attribute] / test32["matches_played"]
    test32 = test32.drop(columns=attribute)

train16.to_csv('data/train16.csv')
train32.to_csv('data/train32.csv')
test16.to_csv('data/test16.csv')
test32.to_csv('data/test32.csv')
