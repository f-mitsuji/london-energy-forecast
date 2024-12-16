from pathlib import Path

import numpy as np
import pandas as pd

from src.settings import INTERIM_WEATHER_DIR, PROCESSED_WEATHER_DIR


def calculate_degree_days(temp: pd.Series, base_temp: float) -> pd.DataFrame:
    diff = temp - base_temp
    degree_days = np.maximum(0, diff)
    degree_days_squared = degree_days**2
    return pd.DataFrame({"linear": degree_days, "squared": degree_days_squared})


def calculate_discomfort_index(temp: pd.Series, humidity: pd.Series) -> pd.Series:
    return 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3


def engineer_weather_features(input_file: Path, output_file: Path, base_temps: dict[str, float]) -> None:
    df = pd.read_csv(input_file, parse_dates=["ob_time"])

    features = pd.DataFrame()
    features["ob_time"] = df["ob_time"]

    cooling_degrees = calculate_degree_days(temp=df["air_temperature"], base_temp=base_temps["cooling"])
    heating_degrees = calculate_degree_days(temp=-df["air_temperature"], base_temp=-base_temps["heating"])

    features["cooling_degree"] = cooling_degrees["linear"].round(1)
    features["cooling_degree_squared"] = cooling_degrees["squared"].round(2)
    features["heating_degree"] = heating_degrees["linear"].round(1)
    features["heating_degree_squared"] = heating_degrees["squared"].round(2)

    features["discomfort_index"] = calculate_discomfort_index(df["air_temperature"], df["rltv_hum"]).round(2)

    features["sun_duration"] = df["wmo_hr_sun_dur"]

    features["cloud_cover"] = (df["cld_ttl_amt_id"] / 9).round(3)

    features.to_csv(output_file, index=False)

    print("\n特徴量の基本統計量:")
    print(features.describe())

    print("\n欠損値の数:")
    print(features.isna().sum())


if __name__ == "__main__":
    base_temps = {
        "cooling": 20.0,
        "heating": 18.0,
    }

    input_file = INTERIM_WEATHER_DIR / "heathrow_weather_2011-2014_linear_interpolated_30min.csv"
    output_file = PROCESSED_WEATHER_DIR / "weather_features_for_power.csv"

    engineer_weather_features(input_file=input_file, output_file=output_file, base_temps=base_temps)
