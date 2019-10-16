# CSU44061 Machine Learning Individual Competition
 Author: Patrick Jennings

 Competition located at https://www.kaggle.com/c/tcdml1920-income-ind/leaderboard

 Kaggle Username: A4C96D6A7369877638D1

## File Layout
The timestamp 'csv's are my 5 submissions that I selected to be used for the private leaderboard of the competition. [15_10_2019_22_49_57.csv](15_10_2019_22_49_57.csv) is my best public leaderboard submission.

[IndivComp.py](IndivComp.py) is the code that runs the prediction. I will talk about the contents of this later.

All the tcd ml.. files are the data provided by the competiton.

## Methodology
As per the comments in [IndivComp.py](IndivComp.py), the following is my methodology:

* Impute values for missing data
    * Most frequent for categorical data
    * Mean for numerical data
* One hot encode categorical data
* Add polynomial features for the categorical data
* Remove outliers from numerical data based on z scores
* Scale the data
* Do MLP regression on the data
* Predict and score the data



