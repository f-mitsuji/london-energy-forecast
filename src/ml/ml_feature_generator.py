import holidays
import numpy as np
import pandas as pd

from src.settings import PROCESSED_DIR, PROCESSED_ENERGY_DIR, PROCESSED_WEATHER_DIR


def _format_hours(periods: float) -> str:
    if periods.is_integer():
        return f"{int(periods)}h"
    return f"{periods:.1f}h"


def is_london_holiday(check_date):
    uk_holidays = holidays.GB(subdiv="ENG")  # type: ignore  # noqa: PGH003
    return int(check_date in uk_holidays)


def load_and_preprocess_data(energy_file, weather_file):
    energy_df = pd.read_csv(energy_file, parse_dates=["DateTime"])
    weather_df = pd.read_csv(weather_file, parse_dates=["ob_time"])

    energy_df = energy_df.rename(columns={"KWH/hh": "demand"})
    energy_df = energy_df.rename(columns={"DateTime": "timestamp"})
    weather_df = weather_df.rename(columns={"ob_time": "timestamp"})

    df = energy_df.merge(weather_df, on="timestamp", how="inner")

    df["time_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["time_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24).round(4)
    df["time_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24).round(4)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).round(4)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).round(4)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

    df["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.map(is_london_holiday)

    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "demand", decimals: int = 3) -> pd.DataFrame:
    lag_periods = [
        1,  # 30分前
        2,  # 1時間前
        3,  # 1時間30分前
        4,  # 2時間前
        48,  # 24時間前
        336,  # 1週間前
    ]
    df_copy = df.copy()

    for lag in lag_periods:
        hours = lag / 2
        col_name = f"{target_col}_lag_{_format_hours(hours)}"
        df_copy[col_name] = df_copy[target_col].shift(lag).round(decimals)

    return df_copy


def add_rolling_features(df: pd.DataFrame, target_col: str = "demand", decimals: int = 3) -> pd.DataFrame:
    windows = {
        "mean": [4, 8, 12],  # 2時間, 4時間, 6時間
        "std": [4, 8],  # 2時間, 4時間
        "min": [4],  # 2時間
        "max": [4],  # 2時間
    }

    df_copy = df.copy()

    for stat, window_sizes in windows.items():
        for window in window_sizes:
            hours = window / 2
            col_name = f"{target_col}_rolling_{_format_hours(hours)}_{stat}"

            roller = df_copy[target_col].rolling(window=window, closed="left")
            if stat == "mean":
                df_copy[col_name] = roller.mean().round(decimals)
            elif stat == "std":
                df_copy[col_name] = roller.std().round(decimals)
            elif stat == "min":
                df_copy[col_name] = roller.min().round(decimals)
            elif stat == "max":
                df_copy[col_name] = roller.max().round(decimals)

    return df_copy


def add_change_features(df: pd.DataFrame, target_col: str = "demand", decimals: int = 3) -> pd.DataFrame:
    change_periods = [
        1,  # 30分
        48,  # 24時間
        336,  # 1週間
    ]
    # change_types = ["diff", "pct"]
    change_types = ["diff"]
    df_copy = df.copy()

    for period in change_periods:
        hours = period / 2
        if "diff" in change_types:
            diff_col = f"{target_col}_diff_{_format_hours(hours)}"
            df_copy[diff_col] = df_copy[target_col].diff(periods=period).round(decimals)

        if "pct" in change_types:
            pct_col = f"{target_col}_pct_change_{_format_hours(hours)}"
            df_copy[pct_col] = df_copy[target_col].pct_change(periods=period).round(decimals)

    return df_copy


def main():
    target_lcl_id = "MAC000145"
    energy_file = PROCESSED_ENERGY_DIR / f"{target_lcl_id}_linear_interpolated.csv"
    weather_file = PROCESSED_WEATHER_DIR / "weather_features_for_power.csv"

    df = load_and_preprocess_data(energy_file, weather_file)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    # df = add_change_features(df)

    df = df.drop(
        columns=[
            # "timestamp",
            "time_of_day",
            "day_of_week",
            "month",
            # "cooling_degree_squared",
            # "heating_degree_squared",
            # "discomfort_index",
        ]
    )

    output_file = PROCESSED_DIR / f"{target_lcl_id}_ml_ready.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
