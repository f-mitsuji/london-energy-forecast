import pandas as pd

from src.settings import INTERIM_ENERGY_DIR, PROCESSED_ENERGY_DIR


def process_power_data(input_file, output_file):
    df = pd.read_csv(
        input_file,
        parse_dates=["DateTime"],
        index_col="DateTime",
    )

    df_processed = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="30min")).interpolate(
        method="linear", limit_direction="both"
    )

    df_processed = df_processed.reset_index().rename(columns={"index": "DateTime"})
    df_processed.to_csv(output_file, index=False, date_format="%Y-%m-%d %H:%M:%S")

    return df_processed


if __name__ == "__main__":
    target_lcl_id = "MAC000145"
    process_power_data(
        input_file=f"{INTERIM_ENERGY_DIR}/{target_lcl_id}.csv",
        output_file=f"{PROCESSED_ENERGY_DIR}/{target_lcl_id}_linear_interpolated.csv",
    )
