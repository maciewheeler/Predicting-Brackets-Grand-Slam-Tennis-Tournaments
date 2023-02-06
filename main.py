from bs4 import BeautifulSoup
import requests
import pandas as pd

links = [
    "1st-serve",
    "1st-serve-points-won",
    "2nd-serve-points-won",
    "service-games-won",
    "1st-serve-return-points-won",
    "break-points-saved",
    "2nd-serve-return-points-won",
    "break-points-converted",
    "return-games-won",
    "aces",
]


for link in links:
    # page = requests.get(f'https://www.atptour.com/en/stats/{link}/2019/hard/all/')
    page = requests.get(f'https://www.atptour.com/en/stats/{link}/2018/hard/all/')
    soup = BeautifulSoup(page.content, 'html.parser')

    tables = soup.select('#statsListingTableContent > table')[0]


    for table in tables:
        
        head = table.text.strip().split('\n')
        head = [x for x in head if x != '']
        if 'Player' in head:
            df = pd.DataFrame(columns=head)
            row_len = len(head) + 1
            cols = head
            print(head)
        
        if len(head) != 0 and 'Player' not in head:

            for idx in range(0, len(head), row_len):
                row = head[idx+1:idx+row_len]
                df.loc[len(df)] = row

            # df.to_csv(f'./data/{link}-2019.csv', index=False)
            df.to_csv(f'./data/{link}-2018.csv', index=False)

        

