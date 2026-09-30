from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_dataset():
    config = load_config()
    path = Path(config["paths"]["raw_data"])

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_excel(path)
    return df, config