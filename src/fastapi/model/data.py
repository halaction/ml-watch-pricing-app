import re
import requests
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from model.paths import CACHE_DIR


def load_data(cfg):

    print("started loading data")

    raw_data = get_raw_data(cfg)

    data = preprocess(cfg, raw_data)
    data_train, data_test = split(cfg, data)
    data_train, data_test = transform(cfg, data_train, data_test)

    data_train.to_pickle(CACHE_DIR / "data-train.pkl")
    data_test.to_pickle(CACHE_DIR / "data-test.pkl")

    print("finished loading data")


def get_raw_data(cfg: DictConfig) -> pd.DataFrame:

    path = CACHE_DIR / "watches (cleaned).csv"

    if not path.exists():

        url = "https://www.kaggle.com/api/v1/datasets/download/beridzeg45/watch-prices-dataset"
        response = requests.get(url, verify=False)
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as file:
            file.extractall(CACHE_DIR)

    raw_data = pd.read_csv(path)

    return raw_data


def preprocess(cfg: DictConfig, raw_data: pd.DataFrame) -> pd.DataFrame:

    data = raw_data.copy()

    print(f"{len(data)=}")

    columns = {column: snake_case(column) for column in data.columns}
    data = data.rename(columns=columns)

    float_columns = [
        "price",
        "water_resistance",
        "face_area",
        "watches_sold_by_the_seller",
        "active_listing_of_the_seller",
        "seller_reviews",
        "year_of_production",
    ]
    categorical_columns = [
        "brand",
        "movement",
        "case_material",
        "bracelet_material",
        "condition",
        "scope_of_delivery",
        "gender",
        "availability",
        "shape",
        "crystal",
        "dial",
        "bracelet_color",
        "clasp",
    ]
    boolean_columns = [
        "fast_shipper",
        "trusted_seller",
        "punctuality",
    ]

    input_columns = float_columns + categorical_columns + boolean_columns

    for column in input_columns:
        assert column in data.columns, f"{column} not in data"

    data = data[input_columns]

    dtypes = {
        **{column: float for column in float_columns},
        **{column: "category" for column in categorical_columns},
        **{column: bool for column in boolean_columns},
    }
    data = data.astype(dtypes)

    optional_columns = [
        "movement",
        "case_material",
        "bracelet_material",
        "gender",
        "shape",
        "crystal",
        "dial",
        "bracelet_color",
        "clasp",
        "availability",
    ]

    for column in categorical_columns:
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

    # data["face_area"] = np.clip(data["face_area"], cfg.min_face_area, cfg.max_face_area)
    data = data[data["face_area"].between(cfg.min_face_area, cfg.max_face_area)]
    data = data[data["price"].between(cfg.min_price, cfg.max_price)]

    data = data.dropna()
    data = data.drop_duplicates()

    print(f"{len(data)=}")

    return data


def snake_case(text: str) -> str:

    text = text.replace(" ", "_").lower()
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)

    return text


def split(cfg: DictConfig, data: pd.DataFrame) -> tuple[pd.DataFrame]:

    data_train, data_test = train_test_split(
        data,
        test_size=cfg.test_size,
        shuffle=True,
        random_state=0,
    )

    return data_train, data_test


def transform(
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

    # brands_count = brands["count"]
    # brands["popularity"] = (brands_count - brands_count.min()) / (
    #     brands_count.max() - brands_count.min()
    # )

    brands["tier"] = None
    brands_price = brands["price_median"]
    quantiles = np.quantile(
        brands_price,
        q=np.linspace(0, 1, cfg.n_brand_tiers + 1),
    )
    for i in range(cfg.n_brand_tiers):
        brands.loc[brands_price.between(quantiles[i], quantiles[i + 1]), "tier"] = i

    brands["range"] = brands["price_max"] / brands["price_min"]

    infrequent_brands = brands[brands["count"] < cfg.min_brand_count].index.tolist()

    # Transform stage
    data_train["brand_count"] = data_train["brand"].map(brands["count"])
    data_test["brand_count"] = data_test["brand"].map(brands["count"])

    # data_train["brand_popularity"] = data_train["brand"].map(brands["popularity"])
    # data_test["brand_popularity"] = data_test["brand"].map(brands["popularity"])

    data_train["brand_tier"] = data_train["brand"].map(brands["tier"])
    data_test["brand_tier"] = data_test["brand"].map(brands["tier"])

    data_train["brand_range"] = data_train["brand"].map(brands["range"])
    data_test["brand_range"] = data_test["brand"].map(brands["range"])

    print(f"{len(data_train['brand'].unique())=}")

    data_train.loc[data_train["brand"].isin(infrequent_brands), "brand"] = "Infrequent"
    data_test.loc[data_test["brand"].isin(infrequent_brands), "brand"] = "Infrequent"

    print(f"{len(data_train['brand'].unique())=}")

    return data_train, data_test
