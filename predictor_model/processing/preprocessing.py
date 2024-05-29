from predictor_model.config import config
import pandas as pd
import numpy as np
import re


def _get_average_salary(df):
    df["salary"] = df[["max_salary", "med_salary", "min_salary"]].mean(axis=1)
    return df


def _convert_to_annual_salary(df):
    df.loc[df["pay_period"] == "MONTHLY", "salary"] = df["salary"] * 12
    df.loc[df["pay_period"] == "WEEKLY", "salary"] = df["salary"] * 52
    df.loc[df["pay_period"] == "HOURLY", "salary"] = df["salary"] * 40 * 52


def _remove_salary_na(df):
    df.dropna(subset=["salary"], inplace=True)
    return df


def _remove_outliers(df):
    mean = df["salary"].mean()
    std = df["salary"].std()
    minimum = 10000  # setting this arbitrarily
    maximum = mean + 2 * std
    df = df[(df["salary"] < maximum) & (df["salary"] > minimum)]
    return df


def _log_transform(df):
    df["salary"] = np.log(df["salary"])
    return df


def _drop_columns(df, drop_features):
    df.drop(columns=drop_features, inplace=True)
    return df


def preprocess_salary(df, drop_features=config.DROP_FEATURES):
    """
    Takes in a dataframe and returns another dataframe with processed salary information.
    input dataframe should have columns "max_salary", "med_salary", "min_salary" and "pay_period"
    Preprocessing includes:
        - converting salary ranges to a single value (mean)
        - converting weekly, hourly and monthly salaries to yearly
        - removing rows with no salary information
        - removing outliers
        - log transformation
        - drop columns (by default, those defined in config)

    Returns a df with columns "description" and "salary"
    """
    df = _get_average_salary(df)
    df = _convert_to_annual_salary(df)
    df = _remove_salary_na(df)
    df = _remove_outliers(df)
    df = _log_transform(df)
    df = _drop_columns(df, drop_features)
    return df


def _remove_salary_info(text):
    text = re.sub("\$\d*\,?\d*", "", text)
    text = re.sub("\d*\,?\d*\$", "", text)
    text = re.sub("(usd|USD)\d*\,?\d*", "", text)
    text = re.sub("\d*\,?\d*(USD|usd)", "", text)
    text = re.sub("\d+(k|K)", "", text)
    text = re.sub("\d+[,|.]\d+", "", text)
    return text


def preprocess_description(df):
    """
    Takes in a dataframe and returns another dataframe with processed description information.
    input dataframe should have column "description"
    Preprocessing includes:
        - removing salary information from text
    Returns a df with columns "description" and any other columns that were part of the input df
    """
    df["description"] = df["description"].map(_remove_salary_info)
    return df
