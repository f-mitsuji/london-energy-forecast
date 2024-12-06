from pathlib import Path

import pandas as pd

from src.settings import INTERIM_WEATHER_DIR


def load_and_combine_data(data_dir: Path, files: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(data_dir / file) for file in files], ignore_index=True)
    df["ob_time"] = pd.to_datetime(df["ob_time"])
    return df.sort_values("ob_time")


def round_weather_values(df: pd.DataFrame) -> pd.DataFrame:
    df["air_temperature"] = df["air_temperature"].round(1)
    df["rltv_hum"] = df["rltv_hum"].round(1)
    df["wmo_hr_sun_dur"] = df["wmo_hr_sun_dur"].round(1)
    df["wetb_temp"] = df["wetb_temp"].round(1)
    df["dewpoint"] = df["dewpoint"].round(1)

    df["cld_ttl_amt_id"] = df["cld_ttl_amt_id"].round(0).astype(int)
    df["visibility"] = ((df["visibility"] / 100).round() * 100).astype(int)
    df["wind_speed"] = df["wind_speed"].round(1).astype(int)
    df["wind_direction"] = ((df["wind_direction"] / 10).round() * 10).astype(int)

    return df


def interpolate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "air_temperature",
        "rltv_hum",
        "wmo_hr_sun_dur",
        "wind_speed",
        "visibility",
        "dewpoint",
        "wetb_temp",
    ]

    df[numeric_columns] = df[numeric_columns].interpolate(method="linear", limit_direction="both")

    df["wind_direction"] = df["wind_direction"].interpolate(method="linear", limit_direction="both")
    df["cld_ttl_amt_id"] = df["cld_ttl_amt_id"].interpolate(method="linear", limit_direction="both")

    return df


def process_weather_data(data_dir: Path, input_files: list[str], output_file: str) -> None:
    df = load_and_combine_data(data_dir, input_files)
    time_range = pd.date_range(start=df["ob_time"].min(), end=df["ob_time"].max(), freq="30min")
    df = df.drop_duplicates(subset=["ob_time"]).set_index("ob_time").reindex(time_range)

    df = interpolate_weather_data(df)
    df = round_weather_values(df)

    df = df.reset_index().rename(columns={"index": "ob_time"})
    df.to_csv(data_dir / output_file, index=False, date_format="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    files = [f"heathrow_{year}.csv" for year in range(2011, 2015)]
    process_weather_data(
        data_dir=INTERIM_WEATHER_DIR,
        input_files=files,
        output_file="heathrow_weather_2011-2014_linear_interpolated_30min.csv",
    )
