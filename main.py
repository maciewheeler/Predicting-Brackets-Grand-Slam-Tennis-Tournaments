# import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd

# link for fetching list of players
link = 'https://www.atptour.com/en/scores/archive/us-open/560/2019/results'
page = requests.get(link)
soup = BeautifulSoup(page.content, 'html.parser')


def player_link(page_ref: str, year:int = 2018) -> str:
    return f"https://www.atptour.com{page_ref}player-stats?year={year}&surfaceType=hard"


main_table = soup.select(
    f"#scoresResultsContent > .day-table-wrapper > .day-table > tbody:nth-of-type(7)")

players = dict()

# store in a dictionary
for tag in main_table[0].find_all('a'):
    if tag.get('href') and 'overview' in tag.get('href'):
        players[tag.text] = tag.get("href")[:-8]

# create function to web-scrap each player's individual stat
def get_player_data(name: str, link: str, year: int = 2018):
    # get link and fetch page
    page = requests.get(player_link(link, year))
    soup = BeautifulSoup(page.content, 'html.parser')
    player_tbls = soup.select("#playerMatchFactsContainer > table > tbody")

    # scrap through both tables and create dataframe
    player_data = dict()
    player_data['name'] = [name]
    player_data['year'] = [year]
    for table in player_tbls:
        for col in table.find_all('tr'):

            first_clean = col.text.strip().split('\n')
            first_clean = [text.strip('\r').strip(
                '\t') for text in first_clean if text.strip('\r').strip('\t')]
            try:
                value = int(first_clean[1])
            except ValueError:
                value = first_clean[1]
            player_data[first_clean[0].replace(' ', '_').lower()] = [value]

    return pd.DataFrame.from_dict(player_data)


data = pd.DataFrame()
# join all player's stats
for name in players:
    data = pd.concat([data, get_player_data(name, players[name], 2018)])

data = data.reset_index(drop=True)
data.to_csv('./data/atp_tour_2018.csv', index=False)
