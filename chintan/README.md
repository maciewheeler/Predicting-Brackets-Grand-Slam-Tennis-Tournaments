# Match to Match Prediction

## How to run the pipeline

When training over the years `2016` to `2018` for testing on the year `2019`

1. Check that these files exist in the data folder

  - [x] aus-open-player-stats-2015.csv
  - [x] aus-open-player-stats-2016.csv
  - [x] aus-open-player-stats-2017.csv
  - [x] m2016.csv
  - [x] m2017.csv
  - [x] m2018.csv
  - [x] m2019.csv

2. Open the notebook `neural_network.ipynb`

3. Check that the years are correctly mentioned.

4. Run the notebook until a preview of the variable `r_16_merge` is displayed.

5. Compare it to the tournament result

### When adding more data,

- Download match outcome data from this [website](http://www.tennis-data.co.uk/ausopen.php)
- If adding data for the year 2015, Save it as `m2015.csv`
- Run the main.py as
```bash
python3 main.py --year 2014
```
- Continue from `Step 2`

## Model used

-   Neural Network

## Assumptions

-   People with no stats have been considered as walkovers
