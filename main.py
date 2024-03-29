# import libraries
from typing import Tuple
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame

# change this for a stat of different year
YEAR = 2018

# link for fetching list of players
link = f'https://www.atptour.com/en/scores/archive/australian-open/580/{YEAR + 1}/results'
page = requests.get(link)
soup = BeautifulSoup(page.content, 'html.parser')

# link for test data
link_test = f'http://www.tennis-data.co.uk/{YEAR + 1}/ausopen.csv'

def player_link(page_ref: str, year: int = YEAR) -> str:
    """Return url for the player stats

    Args:
        page_ref (str): player name and their assigned ATP id
        year (int, optional): year of stats required. Defaults to YEAR.

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


def matches_link(player_ref: str, year: int = YEAR) -> str:
    """returns link of a player's match activity

    Args:
        player_ref (str): player name and their assigned ATP id
        year (int, optional): year of stats required. Defaults to YEAR.

    Returns:
        str: link of the player's activity for YEAR
    """
    return f"https://www.atptour.com{player_ref}player-activity?year={year}&matchType=Singles"


main_table = soup.select(
    f"#scoresResultsContent > .day-table-wrapper > .day-table > tbody:nth-of-type(7)")

players = dict()

# store in a dictionary
for tag in main_table[0].find_all('a'):
    if tag.get('href') and 'overview' in tag.get('href'):
        players[tag.text] = tag.get("href")[:-8]

# create function to web-scrap each player's individual stats, ranking, and height


def get_player_activity(page_ref: str, year: int = YEAR) -> Tuple[str, int]:
    """finds the number of matches played and the height of given player

    Args:
        player_ref (str): player name and their assigned ATP id
        year (int, optional): year of stats required. Defaults to YEAR.

    Returns:
        Tuple[str, int]: a tuple of height and matches played by the player
    """
    match_page = requests.get(matches_link(page_ref, year))
    match_soup = BeautifulSoup(match_page.content, 'html.parser')
    match_data = match_soup.select(".table-height-cm-wrapper")
    height = match_data[0].text[1:-1]
    height = height.strip('cm')
    try:
        height = int(height)
    except ValueError:
        pass
    

    match_data = match_soup.select(".activity-tournament-table")
    matches_played = 0
    for match in match_data:
        tourna_details = match.select('.info-area')
        court = tourna_details[1].text.strip()
        if court.endswith('Hard'):
            a_tournament = match.select('.mega-table > tbody > tr')
            matches_played += len(a_tournament)

    return height, matches_played


def get_player_data(name: str, link: str, year: int = YEAR) -> DataFrame:
    """web-scrap each player's individual stat and ranking

    Args:
        name (str): player's name
        link (str): player's link of statistics
        year (int, optional): year of stats required. Defaults to YEAR.

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
    # loop through and find out the date with the earliest ranking of Jan YEAR
    for date in rank_data:
        if f'{year + 1}.01.' in date.text:
            counter += 1
        if counter == 3:
            singles_rank = date.select('td')[1].text.strip('\n').strip()
            print(f'player: {name}, rank: {singles_rank}')
            player_data['rank'] = singles_rank
            counter = 0
    
    # find the player's height and matches played
    player_acvy = get_player_activity(link)
    player_data['height (cm)'] = player_acvy[0]
    player_data['matches_played'] = player_acvy[1]

    return pd.DataFrame.from_dict(player_data)


data = pd.DataFrame()
# join all player's stats
for name in players:
    data = pd.concat([data, get_player_data(name, players[name], YEAR)])

data = data.reset_index(drop=True)
data.to_csv(f'./data/aus-open-player-stats-{YEAR}.csv', index=False)

test = pd.read_csv(link_test)
test.to_csv(f'./data/aus-open-{YEAR+1}.csv', index=False)
