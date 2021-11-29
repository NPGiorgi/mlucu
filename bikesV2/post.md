# Objective

I want to be able to predict the demand of bike rentals based on historical demand data including climate and date information. We will tackle the challenge as a regression problem. In this case we'll be using Random Forest to create the predictive model.

# Algorithm 

## Decision Trees

-- Place Tree Image Here -- 

Decision Trees are non-parametric supervised learning methods that can be used for both classification and regression. The idea is to be able to create a set of simple rules (from the features), that together, can predict the value of the target variable. 

## Ensemble Methods

The goal behind ensemble methods is to combine the predictions of several estimators in order to provide a more fitting prediction of the target variable. Using ensemble methods usually improves generalization over a single estimator. There are two kinds of ensemble methods:

* Averaging: The principle is having several estimators (built separately) predict the target variable, and then return the average value of all of them.
* Boosting: Attempts to create a strong classifier from a number of weak classifiers.

## Random Forest

Ensemble method that builds decision trees based on samples draws with replacement from the training set. For this problem we will be using particularly Random Forest Regressor (RFR), which will return an averaged value of the values returned by each tree.

# Dataset

The (dataset)[https://www.kaggle.com/c/bike-sharing-demand] contains information about the amount of rentals for a given day, along with climate information for that day.

The dataset has the following features:

* Datetime: hourly date + timestamp
* Season:
	* 1 = spring
	* 2 = summer
	* 3 = fall
	* 4 = winter
* Holiday
* Workingday
* Weather:
	* 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	* 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	* 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	* 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
* Temp: temperature in Celsius
* Atemp: "feels like" temperature in Celsius
* Humidity: relative humidity
* Windspeed: wind speed in km/h
* Casual: number of non-registered user rentals initiated
* Registered: number of registered user rentals initiated
* Count: number of total rentals

`Count` will be our target variable. We'll remove `Casual` and `Register` count as `Casual + Registered = Count`.

# Technology

For this project we'll use mainly [scikit-learn](https://scikit-learn.org/), the code is available at [Github](https://github.com/NPGiorgi/mlucu/blob/main/bikesV2/bikes).

# Data Cleanup

Let's start by loading the dataset and check for missing values.

```python
import pandas as pd

df = pd.read_csv("/path/to/bike_sharing.csv")
df.isnull().sum()
```

| feature    | missing |
| -------    | ------- |
| datetime   |    0    |
| season     |    0    |
| holiday    |    0    |
| workingday |    0    |
| weather    |    0    |
| temp 	   |    0    |
| atemp 	   |    0    |
| humidity   |    0    |
| windspeed  |    0    |
| count      |    0    |

We don't have missing values, so we can proceed to look into the data with more depth. For instance we have the  feature `datetime`, which contains a lot of information into a single field. Let's split that field into separate value for `year`, `month`, `hour` and `day of the week`. 

```python
df['datetime'] = pd.to_datetime(df['datetime'])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df.drop("datetime", axis=1, inplace=True)
```

Our dataset contains now several categorical features. We cannot use them as they come so we'll have to encode them. We'll use `TargetEncoding` for this task. After the categorical data is encoded we'll apply the `StandardScaler` so all values within the dataset are scaled and within a certain range.

```python
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

# encode categorical values into reals using target encoder
df = TargetEncoder(cols=["season", "weather", "hour_of_day", "day_of_week", "month"]).fit_transform(df, df["count"])

# scale features
cols = df.columns.difference(['count'])
df[cols] = StandardScaler().fit_transform(df[cols])
```

# Principal Component Analysis (PCA)

("PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.")[https://scikit-learn.org/stable/modules/decomposition.html#exact-pca-and-probabilistic-interpretation]

In this case we'll use PCA for dimensionality reduction. PCA will give us the variance introduced by each feature, and we'll choose the features that introduce more variance. 

You can check the details on how this is implemented in the repository linked above.

After running PCA, we'll get the following variance per feature:


| Variance | Feature        |  Introduced variance    |
| -------- | -------------- | ----------------------- |
| 0,013942 | workingday     | 1,69582  |
| 0,077469 | windspeed      | 0,345563 |
| 1,399276 | day\_of\_week  | 0,296702 |
| 0,374171 | weather        | 0,282314 |
| 0,773409 | month          | 0,208203 |
| 0,981611 | humidity       | 0,168523 |
| 1,008089 | hour\_of\_day  | 0,139095 |
| 0,465792 | temp           | 0,108872 |
| 1,116961 | holiday        | 0,09162  |
| 1,744838 | day\_of\_month | 0,063528 |
| 0,604886 | season         | 0,026477 |
| 3,440658 | atemp          | 0,013942 |

We'll keep the top seven features (from 1.1 and above), which would be: `workingday`, `windspeed`, `day_of_week`, `weather`, `month`, `humidity`, `hour_of_day` and `temp`. Why from 1.1 and above? I tried training the model several times, and for the features were the introduced variance is 0.11 or below we either don't get a better test score or the training score shows sings of overfitting.

# Correlation Matrix

Now we'll check the correlation values for the remaining non-categorical features. Most of correlated attributes should have been spotted by PCA, but we have double check by looking at the correlation matrix.

-- Image of correlation matrix -- 

We can see a high correlation between `humidity` and `windspeed`. Let's remove humidity in order to avoid overfitting. I actually tried training using `humidity` and I got a better test score, but a much higher training score, which hints to overfitting the training dataset.

# Training the model

We'll split our test in training data and testing data (90/10, because the decision trees are also trained using different data samples). We'll first train using the default parameters for the RFR.


## Default Params

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

cols = df.columns.difference(['count'])

X_train, X_test, y_train, y_test = train_test_split(df[cols], df["count"], test_size=0.10, shuffle=True, random_state=42)

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

predicted_y_train = rfr_grid_search.predict(X_train)
predicted_y_test = rfr_grid_search.predict(X_test)
```

## Scoring

For scoring we'll use `Mean Squared Logarithmic Error`. ("The mean_squared_log error function computes a risk metric corresponding to the expected value of the squared logarithmic (quadratic) error or loss.")[https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error]

```python
print(f"Train score: {mean_squared_log_error(y_true=y_train,y_pred=predicted_y_train)}")
print(f"Test score: {mean_squared_log_error(y_true=y_test,y_pred=predicted_y_test)}")
```

This prints out a score of (usually the lesser the better):

```text
Train score: 0.056718846285641535
Test score: 0.2146775672180868
```

## Hyperparams Search

The RFR algorithm has several parameters we can use, in order to optimize we'll use `HalvingRandomSearchCV`. Successive halving is like a tournament amount candidate parameter combinations. Only some of them make it to the next round, the best is the one remaining at the end of this process.

```python
from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingRandomSearchCV

clf = RandomForestRegressor()

param_distributions = {
    "max_depth": [None] + [i for i in range(3, 100, 2)],
    "min_samples_split": [i for i in range(2, 50, 3)],
    "min_samples_leaf": [i for i in range(2, 50, 3)],
    "max_features": [i for i in range(2, len(df[cols].columns.values))]
}

search = HalvingRandomSearchCV(
    clf,
    param_distributions,
    scoring=make_scorer(mean_squared_log_error, greater_is_better=False),
    resource="n_estimators",
    max_resources=1000,
    cv=3,
    random_state=45,
    n_jobs=-1,
).fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
best_estimator = search.best_estimator_
```

With the model training using the best estimator, we score:

```text
Train score: 0.14512076873370455
Test score: 0.20429924418572723
```

We can see a small improvement in the test score, but a worse score in the training data. This means that we have a more generic and less overfitted model.

If we look at how our model fits the training data before and after:

-- Photo training model before --

-- Photo training model after --

The blue line in the middle represents the case were the target value is exactly the same value as the predicted. We can see that in the model trained without optimizing the hyper parameters, the predicted values are very much around the blue line, which accounts for very accurate predictions. But on the train data, and considering a lower test score, it accounts for overfitting. When we look at the second chart we can see that the values are in a much wider range, which means a worst prediction in general, but in this case it means a more generic model.

If we now look at the predicted values vs. the actual values in the following chart:

-- Photo test model --

We can see the model performs very accurately for higher values. Its performance decreases while the target value decreases.


# Conclusions

This project allowed to test how good ensemble methods are, just running with the default parameters make a good score, but with room for improvement. That's where the hyper parameter search comes in, and with just a few minutes of searching for values we get a much better model for our data. PCA was also a key part of the project, it allowed us to see through the data and really select the features that gives the most benefit to our model, and without it we would have to invest more time in analyzing the underlying relationships in the more carefully.



