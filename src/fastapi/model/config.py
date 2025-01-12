from hydra import compose, initialize
from omegaconf import DictConfig


def get_config(
    config_name: str = "default",
    overrides: list | None = None,
) -> DictConfig:

    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path="config"):
        config = compose(config_name=config_name, overrides=overrides)

    return config
