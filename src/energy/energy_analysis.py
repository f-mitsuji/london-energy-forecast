from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.settings import ENERGY_REPORTS_DIR, INTERIM_ENERGY_DIR


def analyze_energy_data(target_lcl_id, input_dir, report_dir):
    file_path = input_dir / f"{target_lcl_id}.csv"
    df = pd.read_csv(file_path, parse_dates=["DateTime"])

    output_dir = report_dir / target_lcl_id
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_report = {
        "total_records": len(df),
        "date_range": f"{df['DateTime'].min()} to {df['DateTime'].max()}",
        "missing_values": df["KWH/hh"].isna().sum(),
        "unique_dates": df["DateTime"].nunique(),
    }

    time_diffs = df["DateTime"].diff().dropna()
    expected_diff = timedelta(minutes=30)
    irregular_intervals = time_diffs[time_diffs != expected_diff]

    if len(irregular_intervals) > 0:
        quality_report["interval_check"] = "Irregular time intervals detected"
        quality_report["irregular_intervals"] = irregular_intervals.to_dict()
    else:
        quality_report["interval_check"] = "Regular time intervals"

    stats = df["KWH/hh"].describe()
    quality_report["statistics"] = stats.to_dict()

    mean = df["KWH/hh"].mean()
    std = df["KWH/hh"].std()
    outliers = df[abs(df["KWH/hh"] - mean) > 3 * std]
    quality_report["outliers_count"] = len(outliers)

    df["Date"] = df["DateTime"].dt.date
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.day_name()
    df["WeekNumber"] = df["DateTime"].dt.isocalendar().week
    df["Month"] = df["DateTime"].dt.month
    df["MonthName"] = df["DateTime"].dt.strftime("%B")

    plt.figure(figsize=(12, 6))
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    available_months = df["MonthName"].unique()
    month_order = [m for m in month_order if m in available_months]

    sns.boxplot(data=df, x="MonthName", y="KWH/hh", order=month_order)
    plt.title(f"Monthly Energy Consumption Distribution - {target_lcl_id}")
    plt.xlabel("Month")
    plt.ylabel("Energy Consumption (KWH/hh)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    monthly_boxplot_path = output_dir / f"{target_lcl_id}_monthly_boxplot.png"
    plt.savefig(monthly_boxplot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    monthly_avg = df.groupby("MonthName")["KWH/hh"].mean().reindex(month_order)
    plt.plot(monthly_avg.index, monthly_avg.values, marker="o", linewidth=2)
    plt.title(f"Average Monthly Energy Consumption - {target_lcl_id}")
    plt.xlabel("Month")
    plt.ylabel("Average Energy Consumption (KWH/hh)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    monthly_avg_path = output_dir / f"{target_lcl_id}_monthly_average.png"
    plt.savefig(monthly_avg_path)
    plt.close()

    plt.figure(figsize=(15, 8))
    hourly_month = df.pivot_table(values="KWH/hh", index="Hour", columns="MonthName", aggfunc="mean")[month_order]
    sns.heatmap(hourly_month, cmap="YlOrRd", annot=True, fmt=".3f", cbar_kws={"label": "Average KWH/hh"})
    plt.title(f"Average Energy Consumption by Hour and Month - {target_lcl_id}")
    plt.xlabel("Month")
    plt.ylabel("Hour of Day")
    plt.tight_layout()
    monthly_heatmap_path = output_dir / f"{target_lcl_id}_monthly_hourly_heatmap.png"
    plt.savefig(monthly_heatmap_path)
    plt.close()

    plt.figure(figsize=(15, 6))
    daily_consumption = df.groupby("Date")["KWH/hh"].sum().reset_index()
    plt.plot(daily_consumption["Date"], daily_consumption["KWH/hh"], linewidth=1)
    plt.title(f"Daily Energy Consumption - {target_lcl_id}")
    plt.xlabel("Date")
    plt.ylabel("Daily Energy Consumption (KWH)")
    plt.grid(True)
    plt.tight_layout()
    daily_plot_path = output_dir / f"{target_lcl_id}_daily_consumption.png"
    plt.savefig(daily_plot_path)
    plt.close()

    plt.figure(figsize=(15, 6))
    df["WeekStart"] = df["DateTime"].dt.to_period("W").dt.start_time
    weekly_consumption = df.groupby("WeekStart")["KWH/hh"].mean().reset_index()
    plt.plot(weekly_consumption["WeekStart"], weekly_consumption["KWH/hh"], marker="o", linewidth=2, color="darkblue")
    plt.title(f"Weekly Average Energy Consumption - {target_lcl_id}")
    plt.xlabel("Week Starting Date")
    plt.ylabel("Average Energy Consumption (KWH/hh)")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    weekly_plot_path = output_dir / f"{target_lcl_id}_weekly_consumption.png"
    plt.savefig(weekly_plot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=df, x="DayOfWeek", y="KWH/hh", order=day_order)
    plt.title(f"Energy Consumption by Day of Week - {target_lcl_id}")
    plt.xlabel("Day of Week")
    plt.ylabel("Energy Consumption (KWH/hh)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    dayofweek_plot_path = output_dir / f"{target_lcl_id}_dayofweek_consumption.png"
    plt.savefig(dayofweek_plot_path)
    plt.close()

    analysis_report = {
        "月別平均消費電力": df.groupby("MonthName")["KWH/hh"].mean().to_dict(),
        "最も消費電力が多い月": df.groupby("MonthName")["KWH/hh"].mean().idxmax(),
        "最も消費電力が少ない月": df.groupby("MonthName")["KWH/hh"].mean().idxmin(),
        "日別平均消費電力": daily_consumption["KWH/hh"].mean(),
        "週別平均消費電力": weekly_consumption["KWH/hh"].mean(),
        "曜日別平均消費電力": df.groupby("DayOfWeek")["KWH/hh"].mean().to_dict(),
        "時間帯別ピーク消費時刻": df.groupby("Hour")["KWH/hh"].mean().idxmax(),
        "最も消費電力が多い曜日": df.groupby("DayOfWeek")["KWH/hh"].mean().idxmax(),
    }

    report_path = output_dir / f"{target_lcl_id}_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("電力消費データ総合分析レポート\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. データ品質チェック結果\n")
        f.write("-" * 30 + "\n")
        for key, value in quality_report.items():
            f.write(f"{key}:\n{value}\n\n")

        f.write("\n2. 消費電力分析結果\n")
        f.write("-" * 30 + "\n")
        for key, value in analysis_report.items():
            f.write(f"{key}:\n{value}\n\n")

    return quality_report, analysis_report, df


if __name__ == "__main__":
    quality_report, analysis_report, df = analyze_energy_data("MAC000152", INTERIM_ENERGY_DIR, ENERGY_REPORTS_DIR)
