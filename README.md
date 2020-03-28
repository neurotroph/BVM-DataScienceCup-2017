# Submission to BVM Data Science Cup 2017
This repository contains the python scripts I used for my submission to the [BVM Data Science Cup 2017](https://bvm.org/datasciencecup2017/). My solution ranked 3rd - of five submissions, so there is a lot room for improvements (see below).

The code is generally messy and not consistently documented. While I will not work on the code itself, I might add more reasonable comments and docstrings to the code in the next weeks.

## Requirements
See `requirements.txt`, but generally:
* Python 3.6
* `psycopg2`
* [Facebook Prophet](https://github.com/facebook/prophet) library and dependencies

## Task definitions
Basis for the tasks are sales data from the MMO "EVE Online", which were provided on the [BVM website](https://bvm.org/datasciencecup2017/). The data was available as CSV data, which could easily be imported into a PostgreSQL database. Due to the size of the data dump, it is not part of this repository, but you might still be able to download the data from the competition's website.

The tasks were to predict future sales and indicators based on historical data.
1. Predict sales for five products in four product segments: `predict-items.py`
2. Predict three price indices: `predict-indices.py`
3. Predict market shares: `predict-turnovers.py`
4. Give an optimal plan for selling items: `task-4.py`

## Ideas and Rationale for my solution
The available data were rich, but I didn't use its full potential.

My first idea was to build a Bayesian multilevel regression model to not only model the trends in sales over time but also model additional predictors. For example, one can assume that products will have different trends in different regions of space. Based on what (limited) knowledge I have about EVE, I assume that different regions will have different economies, based on factions, available resources etc. Thus, modeling only a time series for the dependent variable (sales) based on it's general, macroscopic trends would be somewhat short-sighted.

However, contraints in time and knowledge didn't allow me to fully go down this road. As Facebook released their `prophet` library only weeks before and [their paper](https://peerj.com/preprints/3190.pdf) was very convincing in terms of predictive accuracy I gave it a try. Most importantly, their library was very fast, so I was able to easily test some code with it.

Unfortunately, I didn't find a very good solution to task 4, so I basically tried to get some basic overview where I could sell the products for a good price and randomly dropped some predictions into the results excel. It was no surprise to me that my solution was the worst for this task. My final predictions are not included in the repository.

## Possible Improvements
I probably will not go back to this code (it's a mess, sorry) and improve it, but there are some things that are missing from this approach and should be included in any further attempt to do something similar. In the mean time (it was almost a year ago...) I have learned a lot and would do things very, very differently. This includes:
* Proper cross-validation, even if only for the parameters of the `prophet` library
* Do more data investigation and visualization before starting to code
* Do some hierarchical modeling in some way or the other (see rationale above)
* Do actual modeling for task 4 (e.g. include actual transactions for the predictions of an optimal selling plan)

Facebook Prophet does something similar to what I had in mind (they actually use Stan!), but obviously cannot take the hierarchical nature of the data into account.
