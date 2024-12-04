import csv
from collections import defaultdict

from src.settings import INTERIM_WEATHER_DIR, RAW_WEATHER_DIR


def get_header_indices(header_file):
    header_dict = {}
    with header_file.open(encoding="utf-8") as f:
        header_line = f.readline().strip()
        headers = header_line.split(",")
        for idx, header in enumerate(headers):
            header_dict[header.strip()] = idx
    return header_dict


def extract_weather(input_file, output_file, src_id):
    header_file = RAW_WEATHER_DIR / "WH_Column_Headers.txt"
    header_indices = get_header_indices(header_file)

    selected_weather_features = [
        "OB_TIME",
        "AIR_TEMPERATURE",
        "RLTV_HUM",
        "WMO_HR_SUN_DUR",
        "WIND_SPEED",
        "CLD_TTL_AMT_ID",
        "VISIBILITY",
        "DEWPOINT",
        "WETB_TEMP",
        "WIND_DIRECTION",
        # "PRST_WX_ID",  # 欠損値多い
    ]

    time_data = defaultdict(list)
    try:
        with input_file.open(encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue

                parts = line.strip().split(",")
                if (
                    src_id == parts[header_indices["SRC_ID"]].strip()
                    and parts[header_indices["MET_DOMAIN_NAME"]].strip() == "SYNOP"
                ):
                    if len(parts) != len(header_indices):
                        print(f"Warning: Invalid data found: {parts}")

                    ob_time = parts[header_indices["OB_TIME"]].strip()

                    row_data = []
                    for feature in selected_weather_features:
                        value = parts[header_indices[feature]].strip()
                        row_data.append(value)

                    time_data[ob_time].append(row_data)

        with output_file.open("w", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([feature.lower() for feature in selected_weather_features])

            for entries in time_data.values():
                if len(entries) == 1:
                    writer.writerow(entries[0])
                else:
                    writer.writerow(entries[-1])

    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except Exception as e:  # noqa: BLE001
        print(f"An error occurred: {e!s}")


if __name__ == "__main__":
    target_years = [2011, 2012, 2013, 2014]
    target_src_id = "708"
    target_src_name = "heathrow"

    for target_year in target_years:
        input_file = RAW_WEATHER_DIR / f"midas_wxhrly_{target_year}01-{target_year}12.txt"
        output_file = INTERIM_WEATHER_DIR / f"{target_src_name}_{target_year}.csv"

        extract_weather(input_file, output_file, target_src_id)
        print(f"Extracted weather ({target_year}) has been saved to {output_file}")
