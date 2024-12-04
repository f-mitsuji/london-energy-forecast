import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.settings import INTERIM_WEATHER_DIR, WEATHER_REPORTS_DIR


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df["ob_time"] = pd.to_datetime(df["ob_time"])
    return df.set_index("ob_time")


def check_missing_values(df):
    missing_stats = pd.DataFrame(
        {"Missing Count": df.isna().sum(), "Missing Percentage": (df.isna().sum() / len(df) * 100).round(2)}
    )
    return missing_stats.sort_values("Missing Percentage", ascending=False)


def check_value_ranges(df):
    numeric_stats = df.describe()

    validity_checks = {
        "air_temperature": (-50, 50),  # 気温 (°C)
        "rltv_hum": (0, 100),  # 相対湿度 (%)
        "wmo_hr_sum_dur": (0, 1),  # 日照時間 (h)
        "wind_speed": (0, 200),  # 風速 (knots)
        "visibility": (0, 100000),  # 視程 (m)
        "dewpoint": (-50, 50),  # 露点温度 (°C)
        "wetb_temp": (-50, 50),  # 湿球温度 (°C)
        "wind_direction": (0, 360),  # 風向 (度)
    }

    out_of_range = {}
    for col, (min_val, max_val) in validity_checks.items():
        if col in df.columns:
            invalid_count = df[(df[col] < min_val) | (df[col] > max_val)].shape[0]
            if invalid_count > 0:
                out_of_range[col] = {
                    "invalid_count": invalid_count,
                    "percentage": round(invalid_count / len(df) * 100, 2),
                    "min_found": df[col].min(),
                    "max_found": df[col].max(),
                }

    return numeric_stats, out_of_range


def plot_time_series(df):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Weather Parameters Time Series", fontsize=16)

    # 気温、露点温度、湿球温度
    ax = axes[0, 0]
    df[["air_temperature", "dewpoint", "wetb_temp"]].plot(ax=ax)
    ax.set_title("Temperatures")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()

    # 相対湿度
    ax = axes[0, 1]
    df["rltv_hum"].plot(ax=ax)
    ax.set_title("Relative Humidity")
    ax.set_ylabel("Humidity (%)")

    # 風速
    ax = axes[1, 0]
    df["wind_speed"].plot(ax=ax)
    ax.set_title("Wind Speed")
    ax.set_ylabel("Speed (knots)")

    # 風向
    ax = axes[1, 1]
    df["wind_direction"].plot(ax=ax, style=".")
    ax.set_title("Wind Direction")
    ax.set_ylabel("Direction (degrees)")

    # 視程
    ax = axes[2, 0]
    df["visibility"].plot(ax=ax)
    ax.set_title("Visibility")
    ax.set_ylabel("Visibility (m)")

    # 雲量
    ax = axes[2, 1]
    df["cld_ttl_amt_id"].plot(ax=ax)
    ax.set_title("Cloud Amount")
    ax.set_ylabel("Cloud Amount ID")

    plt.tight_layout()
    return fig


def plot_distributions(df):
    numeric_cols = [
        "air_temperature",
        "rltv_hum",
        "wind_speed",
        "visibility",
        "dewpoint",
        "wetb_temp",
        "wind_direction",
    ]

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4 * len(numeric_cols)))
    fig.suptitle("Distribution of Weather Parameters", fontsize=16)

    for ax, col in zip(axes, numeric_cols, strict=False):
        if col in df.columns:
            sns.histplot(data=df, x=col, ax=ax)
            ax.set_title(f"Distribution of {col}")

    plt.tight_layout()
    return fig


def check_temporal_consistency(df):
    time_diffs = df.index.to_series().diff().dropna()
    time_stats = time_diffs.describe()

    irregular_intervals = time_diffs[time_diffs != pd.Timedelta(hours=1)]
    irregular_intervals_summary = {
        "count": len(irregular_intervals),
        "details": [
            {"timestamp": str(idx), "interval": str(interval)} for idx, interval in irregular_intervals.items()
        ],
    }

    rapid_changes = {
        "AIR_TEMPERATURE": 10,  # 10°C以上の変化
        "WIND_SPEED": 30,  # 30knots以上の変化
        "VISIBILITY": 50000,  # 50km以上の変化
    }

    anomalies = {}
    for col, threshold in rapid_changes.items():
        if col in df.columns:
            changes = df[col].diff().abs()
            rapid = changes[changes > threshold]
            if len(rapid) > 0:
                anomalies[col] = {"count": len(rapid), "max_change": rapid.max(), "timestamps": rapid.index.tolist()}

    return time_stats, anomalies, irregular_intervals_summary


def generate_weather_report(file_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(file_path)

    missing_stats = check_missing_values(df)
    numeric_stats, out_of_range = check_value_ranges(df)
    time_stats, anomalies, irregular_intervals = check_temporal_consistency(df)

    with (output_dir / "weather_quality_report.txt").open("w") as f:
        f.write("Weather Data Quality Report\n")
        f.write("=========================\n\n")

        f.write("1. Basic Information\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n\n")

        f.write("2. Missing Values\n")
        f.write(missing_stats.to_string())
        f.write("\n\n")

        f.write("3. Value Ranges\n")
        f.write("Statistical Summary:\n")
        f.write(numeric_stats.to_string())
        f.write("\n\nOut of Range Values:\n")
        for col, stats in out_of_range.items():
            f.write(f"\n{col}:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n4. Temporal Consistency\n")
        f.write("Time Interval Statistics:\n")
        f.write(time_stats.to_string())

        f.write("\n\nIrregular Time Intervals:\n")
        f.write(f"Total irregular intervals: {irregular_intervals['count']}\n")
        if irregular_intervals["count"] > 0:
            f.write("Details:\n")
            for detail in irregular_intervals["details"]:
                f.write(f"  Timestamp: {detail['timestamp']}, Interval: {detail['interval']}\n")

        f.write("\n\nRapid Changes Detected:\n")
        for col, stats in anomalies.items():
            f.write(f"\n{col}:\n")
            f.write(f"  Number of rapid changes: {stats['count']}\n")
            f.write(f"  Maximum change: {stats['max_change']}\n")

    time_series_fig = plot_time_series(df)
    time_series_fig.savefig(output_dir / "time_series_plots.png")
    plt.close(time_series_fig)

    dist_fig = plot_distributions(df)
    dist_fig.savefig(output_dir / "distribution_plots.png")
    plt.close(dist_fig)


if __name__ == "__main__":
    target_years = [2011, 2012, 2013, 2014]
    target_src_name = "heathrow"

    for target_year in target_years:
        input_file = INTERIM_WEATHER_DIR / f"{target_src_name}_{target_year}.csv"
        generate_weather_report(input_file, WEATHER_REPORTS_DIR / f"weather_report_{target_year}")
        print(f"Weather quality report ({target_year}) has been generated")
