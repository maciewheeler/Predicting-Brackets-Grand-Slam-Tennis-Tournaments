# import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame

# change this for a stat of different year
YEAR = 2018

# link for fetching list of players
link = 'https://www.atptour.com/en/scores/archive/australian-open/580/2019/results'
page = requests.get(link)
soup = BeautifulSoup(page.content, 'html.parser')


def player_link(page_ref: str, year: int = 2018) -> str:
    """Return url for the player stats

    Args:
        page_ref (str): player name and their assigned ATP id
        year (int, optional): year of stats required. Defaults to 2018.

    Returns:
        str: url with player stats of the given ref and year
    """
    return f"https://www.atptour.com{page_ref}player-stats?year={year}&surfaceType=hard"


def rank_link(page_ref: str) -> str:
    """returns url for history of player's rank

    Args:
        page_ref (str): player name and their assigned ATP id

    Returns:
        str: url of the player's ranking over the years
    """
    return f"https://www.atptour.com{page_ref}rankings-history"


main_table = soup.select(
    f"#scoresResultsContent > .day-table-wrapper > .day-table > tbody:nth-of-type(7)")

players = dict()

# store in a dictionary
for tag in main_table[0].find_all('a'):
    if tag.get('href') and 'overview' in tag.get('href'):
        players[tag.text] = tag.get("href")[:-8]

# create function to web-scrap each player's individual stat and ranking


def get_player_data(name: str, link: str, year: int = 2018) -> DataFrame:
    """web-scrap each player's individual stat and ranking

    Args:
        name (str): player's name
        link (str): player's link of statistics
        year (int, optional): year of stats required. Defaults to 2018.

    Returns:
        DataFrame: returns dataframe of the player's stats and ranking
    """
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

    # requesting rankings history page
    rank_page = requests.get(rank_link(link))
    rank_soup = BeautifulSoup(rank_page.content, 'html.parser')
    # selector for the list of date, single_rank, double_rank
    rank_data = rank_soup.select(
        "#playerRankHistoryContainer > table > tbody > tr")

    counter = 0
    # special case**
    if name == 'Janko Tipsarevic':
        no_rank = float('-inf')
        print(f'player: {name}, rank: {no_rank}')
        player_data['rank'] = float('-inf')

    # loop through and find out the date with the earliest ranking of Jan 2019
    for date in rank_data:
        if '2019.01.' in date.text:
            counter += 1
        if counter == 3:
            singles_rank = date.select('td')[1].text.strip('\n').strip()
            print(f'player: {name}, rank: {singles_rank}')
            player_data['rank'] = singles_rank
            counter = 0

    return pd.DataFrame.from_dict(player_data)


data = pd.DataFrame()
# join all player's stats
for name in players:
    data = pd.concat([data, get_player_data(name, players[name], YEAR)])

data = data.reset_index(drop=True)
data.to_csv(f'./data/aus-open-player-stats-{YEAR}.csv', index=False)


# ** - Janko Tipsarevic doesn't have a ranking because he didn't take part
# in tournaments post 2017 US Open. He got ranking on Jan 28, 2019.
# The earliest ranking before 2019 Aus Open is Aug 27, 2018
