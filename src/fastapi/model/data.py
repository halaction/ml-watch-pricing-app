import re

import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import kagglehub


def get_raw_data(cfg: DictConfig) -> pd.DataFrame:

    path = kagglehub.dataset_download(
        "beridzeg45/watch-prices-dataset",
        path="watches (cleaned).csv",
    )

    raw_data = pd.read_csv(path)

    return raw_data


def get_data(cfg: DictConfig) -> pd.DataFrame:

    raw_data = get_raw_data()
    data = raw_data.copy()

    columns = {column: snake_case(column) for column in data.columns}
    data = data.rename(columns=columns)

    relevant_columns = [
        "brand",
        "movement",
        "condition",
        "scope_of_delivery",
        "water_resistance",
        "face_area",
        "case_material",
        "bracelet_material",
        "gender",
        "shape",
        "crystal",
        "dial",
        "bracelet_color",
        "clasp",
        "fast_shipper",
        "trusted_seller",
        "punctuality",
        "price",
        # 'year_of_production',
        # 'availability',
        # 'watches_sold_by_the_seller',
        # 'active_listing_of_the_seller',
        # 'seller_reviews'
    ]

    for column in relevant_columns:
        assert column in data.columns, f"{column} not in data"

    data = data[relevant_columns]

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
    ]
    data = data.fillna({column: "Undefined" for column in optional_columns})

    data = data.fillna(
        {
            "water_resistance": 0,
            "face_area": 700,
        }
    )

    data = data.dropna()
    data = data.drop_duplicates()

    data = data.astype(
        {
            "fast_shipper": bool,
            "trusted_seller": bool,
            "punctuality": bool,
        }
    )

    return data


def snake_case(text: str) -> str:

    text = text.replace(" ", "_").lower()
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)

    return text


def split_data(cfg: DictConfig, data: pd.DataFrame) -> pd.DataFrame:

    test_size = cfg.test_size
    val_size = cfg.val_size / (1 - test_size)

    data_train, data_test = train_test_split(data, test_size=test_size)
    data_train, data_val = train_test_split(data_train, test_size=val_size)

    return data_train, data_val, data_test
