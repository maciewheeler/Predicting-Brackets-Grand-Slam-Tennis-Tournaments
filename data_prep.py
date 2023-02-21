import pandas as pd

train = pd.read_csv('data/aus-open-player-stats-2018.csv')

count = 0
for i in train["1st_serve"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "1st_serve"] = percent
    count = count + 1

count = 0
for i in train["1st_serve_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "1st_serve_points_won"] = percent
    count = count + 1

count = 0
for i in train["2nd_serve_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "2nd_serve_points_won"] = percent
    count = count + 1

count = 0
for i in train["break_points_saved"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "break_points_saved"] = percent
    count = count + 1

count = 0
for i in train["service_games_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "service_games_won"] = percent
    count = count + 1

count = 0
for i in train["total_service_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "total_service_points_won"] = percent
    count = count + 1

count = 0
for i in train["1st_serve_return_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "1st_serve_return_points_won"] = percent
    count = count + 1

count = 0
for i in train["2nd_serve_return_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "2nd_serve_return_points_won"] = percent
    count = count + 1

count = 0
for i in train["break_points_converted"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "break_points_converted"] = percent
    count = count + 1

count = 0
for i in train["return_games_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "return_games_won"] = percent
    count = count + 1

count = 0
for i in train["return_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "return_points_won"] = percent
    count = count + 1

count = 0
for i in train["total_points_won"]:
    percent = float(i.strip("%")) / 100
    train.loc[:, "total_points_won"] = percent
    count = count + 1

train.to_csv('data/aus-open-player-stats-2018.csv')
