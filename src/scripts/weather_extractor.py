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


def get_quality_value(parts, feature, header_indices):
    # OB_TIME、RLTV_HUMには品質列なし
    if feature in ["OB_TIME", "RLTV_HUM"]:
        return None

    quality_column = f"{feature}_Q"
    if quality_column in header_indices:
        return parts[header_indices[quality_column]].strip()
    return None


def get_qc_level(quality):
    """品質値から末尾の数字(qc_level)を取得."""
    if not quality:
        return -1
    return int(quality[-1])


def extract_weather(input_file, output_file, src_id):
    header_file = RAW_WEATHER_DIR / "WH_Column_Headers.txt"
    header_indices = get_header_indices(header_file)

    selected_weather_features = [
        "OB_TIME",
        "AIR_TEMPERATURE",
        # "CS_HR_SUN_DUR",
        "WMO_HR_SUN_DUR",
        "DEWPOINT",
        "CLD_TTL_AMT_ID",
        "WETB_TEMP",
        "VISIBILITY",
        "WIND_SPEED",
        "WIND_DIRECTION",
        # "PRST_WX_ID",
        "RLTV_HUM",
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
                    row_qualities = []

                    for feature in selected_weather_features:
                        value = parts[header_indices[feature]].strip()
                        quality = get_quality_value(parts, feature, header_indices)
                        row_data.append(value)
                        row_qualities.append(quality)

                    time_data[ob_time].append((row_data, row_qualities))

        with output_file.open("w", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([feature.lower() for feature in selected_weather_features])

            for entries in time_data.values():
                if len(entries) == 1:
                    writer.writerow(entries[0][0])
                else:
                    # 重複がある場合、各エントリーの品質レベルの最小値を比較
                    best_entry = None
                    best_min_qc_level = -1

                    for entry, qualities in entries:
                        # 品質値がある要素のみを考慮
                        valid_qualities = [q for q in qualities if q is not None]
                        qc_levels = [-1] if not valid_qualities else [get_qc_level(q) for q in valid_qualities]

                        # 最も低いqc_levelを取得
                        min_qc_level = min(qc_levels)

                        # より高いqc_levelを持つエントリーを選択
                        if min_qc_level > best_min_qc_level:
                            best_entry = entry
                            best_min_qc_level = min_qc_level

                    writer.writerow(best_entry[0] if best_entry else entries[0][0])

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
