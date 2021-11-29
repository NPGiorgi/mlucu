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

Ensemble method that builds decision trees based on samples draws with replacement from the training set. For this problem we will be using particularly Random Forest Regressor, which will return an averaged value of the values returned by each tree.

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

After running PCA, we'll get the following variance per feature


| Variance | Feature        |  Introduced Variance    |
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

We'll keep the top eight features (from 1.1 and above), which would be: `workingday`, `windspeed`, `day_of_week`, `weather`, `month`, `humidity`, `hour_of_day` and `temp`. Why eight? I tried training the model several times, 

