import holidays
import numpy as np
import pandas as pd

from src.settings import PROCESSED_DIR, PROCESSED_ENERGY_DIR, PROCESSED_WEATHER_DIR


def is_london_holiday(check_date):
    uk_holidays = holidays.GB(subdiv="ENG")  # type: ignore  # noqa: PGH003
    return int(check_date in uk_holidays)


def load_and_preprocess_data(energy_file, weather_file):
    energy_df = pd.read_csv(energy_file, parse_dates=["DateTime"])
    weather_df = pd.read_csv(weather_file, parse_dates=["ob_time"])

    energy_df = energy_df.rename(columns={"DateTime": "timestamp"})
    weather_df = weather_df.rename(columns={"ob_time": "timestamp"})

    df = energy_df.merge(weather_df, on="timestamp", how="inner")

    df["time_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.map(is_london_holiday)

    df.drop(columns=["cooling_degree_squared", "heating_degree_squared"])

    return df.drop(columns=["timestamp", "day_of_week", "month", "time_of_day"])


def create_time_windows(df, target_col="KWH/hh", window_sizes=None):
    if window_sizes is None:
        window_sizes = [2, 6, 12, 24, 48]
    df = df.sort_values("timestamp")

    for window in window_sizes:
        df[f"mean_{window/2}h"] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f"max_{window/2}h"] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f"min_{window/2}h"] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f"std_{window/2}h"] = df[target_col].rolling(window=window, min_periods=1).std()

    return df


def main():
    target_lcl_id = "MAC000145"
    energy_file = PROCESSED_ENERGY_DIR / f"{target_lcl_id}_linear_interpolated.csv"
    weather_file = PROCESSED_WEATHER_DIR / "weather_features_for_power.csv"

    df = load_and_preprocess_data(energy_file, weather_file)
    # df_with_time_features = create_time_windows(df)

    output_file = PROCESSED_DIR / f"{target_lcl_id}_ml_ready.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
