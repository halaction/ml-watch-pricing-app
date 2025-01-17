import re
import requests
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from model.paths import DATA_DIR


def get_raw_data(cfg: DictConfig) -> pd.DataFrame:

    path = DATA_DIR / "watches (cleaned).csv"

    if not path.exists():

        url = "https://www.kaggle.com/api/v1/datasets/download/beridzeg45/watch-prices-dataset"
        response = requests.get(url, verify=False)
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as file:
            file.extractall(DATA_DIR)

    raw_data = pd.read_csv(path)

    return raw_data


def preprocess(cfg: DictConfig, raw_data: pd.DataFrame) -> pd.DataFrame:

    data = raw_data.copy()

    # Rename columns
    columns = {column: snake_case(column) for column in data.columns}
    data = data.rename(columns=columns)

    # Check required columns
    input_columns = (
        cfg.numerical_columns + cfg.categorical_columns + cfg.boolean_columns
    )

    for column in input_columns:
        assert column in data.columns, f"{column} not in data"

    data = data[input_columns]

    # Cast to specified dtypes
    data = data.astype({column: np.float64 for column in cfg.numerical_columns})
    data = data.astype({column: "category" for column in cfg.categorical_columns})
    data = data.astype({column: bool for column in cfg.boolean_columns})

    # Fill missing values
    for column in cfg.categorical_columns:
        data[column] = (
            data[column]
            .cat.add_categories(["Undefined", "Infrequent"])
            .fillna("Undefined")
        )

    data = data.fillna(
        {
            "water_resistance": 0,
            "face_area": 700,
            "watches_sold_by_the_seller": 0,
            "active_listing_of_the_seller": 1,
            "seller_reviews": 0,
        }
    )

    # Add features
    decades = [1980, 1990, 2000, 2010, 2020]
    for decade in decades:
        data[f"produced_before_{decade}"] = np.where(
            data["year_of_production"].isna(),
            False,
            data["year_of_production"] < decade,
        )

    data["produced_in_2024"] = np.where(
        data["year_of_production"].isna(),
        False,
        data["year_of_production"] == 2024,
    )

    data = data.drop(columns=["year_of_production"])

    data["original_box"] = data["scope_of_delivery"].isin(
        [
            "Original box, original papers",
            "Original box, no original papers",
        ]
    )
    data["original_papers"] = data["scope_of_delivery"].isin(
        [
            "Original box, original papers",
            "Original papers, no original box",
        ]
    )

    data = data.dropna()
    data = data.drop_duplicates()

    return data


def enforce_constraints(cfg, data):

    data = data[data["face_area"].between(cfg.min_face_area, cfg.max_face_area)]
    data = data[data["price"].between(cfg.min_price, cfg.max_price)]

    return data


def check_constraints(cfg, data):

    assert data["face_area"].between(cfg.min_face_area, cfg.max_face_area).all()
    assert data["price"].between(cfg.min_price, cfg.max_price).all()


def snake_case(text: str) -> str:

    text = text.replace(" ", "_").lower()
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)

    return text


def split(cfg: DictConfig, data: pd.DataFrame) -> tuple[pd.DataFrame]:

    data_train, data_test = train_test_split(
        data,
        test_size=cfg.test_size,
        # stratify=data["brand"],
        shuffle=True,
        random_state=0,
    )

    return data_train, data_test


def fit_transform(
    cfg: DictConfig,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
) -> tuple[pd.DataFrame]:

    # Fit stage
    brands = data_train.groupby(by="brand", observed=True).agg(
        price_min=pd.NamedAgg(column="price", aggfunc="min"),
        price_max=pd.NamedAgg(column="price", aggfunc="max"),
        price_median=pd.NamedAgg(column="price", aggfunc="median"),
        count=pd.NamedAgg(column="price", aggfunc="count"),
    )

    brands["tier"] = None
    brand_prices = brands["price_median"]

    num = cfg.n_brand_tiers + 1
    q = np.linspace(0, 1, num=num)
    quantiles = np.quantile(brand_prices, q=q)

    for i in range(cfg.n_brand_tiers):
        brands.loc[brand_prices.between(quantiles[i], quantiles[i + 1]), "tier"] = i

    brands["range"] = brands["price_max"] / brands["price_min"]

    brands_path = DATA_DIR / "brands.csv"
    brands.to_csv(brands_path, index=True)

    # Transform stage
    valid_brands = brands.index
    data_train = data_train[data_train["brand"].isin(valid_brands)]
    data_test = data_test[data_test["brand"].isin(valid_brands)]

    data_train["brand_count"] = data_train["brand"].map(brands["count"])
    data_test["brand_count"] = data_test["brand"].map(brands["count"])

    data_train["brand_tier"] = data_train["brand"].map(brands["tier"])
    data_test["brand_tier"] = data_test["brand"].map(brands["tier"])

    data_train["brand_range"] = data_train["brand"].map(brands["range"])
    data_test["brand_range"] = data_test["brand"].map(brands["range"])

    infrequent_brands = brands[brands["count"] < cfg.min_brand_count].index
    data_train.loc[data_train["brand"].isin(infrequent_brands), "brand"] = "Infrequent"
    data_test.loc[data_test["brand"].isin(infrequent_brands), "brand"] = "Infrequent"

    return data_train, data_test


def transform(
    cfg: DictConfig,
    data: pd.DataFrame,
) -> pd.DataFrame:

    brands_path = DATA_DIR / "brands.csv"
    brands = pd.read_csv(brands_path).set_index("brand")

    # Transform stage
    valid_brands = brands.index
    data = data[data["brand"].isin(valid_brands)]

    data["brand_count"] = data["brand"].map(brands["count"])
    data["brand_tier"] = data["brand"].map(brands["tier"])
    data["brand_range"] = data["brand"].map(brands["range"])

    infrequent_brands = brands[brands["count"] < cfg.min_brand_count].index
    data.loc[data["brand"].isin(infrequent_brands), "brand"] = "Infrequent"

    return data
