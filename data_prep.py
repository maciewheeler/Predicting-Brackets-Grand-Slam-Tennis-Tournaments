import pandas as pd

train = pd.read_csv('data/aus-open-player-stats-2018.csv')
test = pd.read_csv('data/aus-open-2019.csv')

# count = 0
# for i in train["1st_serve"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "1st_serve"] = percent
#     count = count + 1
#
# count = 0
# for i in train["1st_serve_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "1st_serve_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["2nd_serve_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "2nd_serve_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["break_points_saved"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "break_points_saved"] = percent
#     count = count + 1
#
# count = 0
# for i in train["service_games_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "service_games_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["total_service_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "total_service_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["1st_serve_return_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "1st_serve_return_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["2nd_serve_return_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "2nd_serve_return_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["break_points_converted"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "break_points_converted"] = percent
#     count = count + 1
#
# count = 0
# for i in train["return_games_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "return_games_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["return_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "return_points_won"] = percent
#     count = count + 1
#
# count = 0
# for i in train["total_points_won"]:
#     percent = float(i.strip("%")) / 100
#     train.loc[count, "total_points_won"] = percent
#     count = count + 1
#
# train.to_csv('data/aus-open-player-stats-2018.csv')

new16 = test[test["Round"] == "4th Round"]
new_test16 = new16[["Winner", "Loser", "WRank", "LRank"]]

new_test16.to_csv("data/aus-open-2019-top-16.csv")
