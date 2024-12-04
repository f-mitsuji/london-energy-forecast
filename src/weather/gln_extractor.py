from datetime import datetime

from src.settings import INTERIM_WEATHER_DIR, RAW_WEATHER_DIR


def extract_gln_records(input_file, output_file):
    try:
        with input_file.open(encoding="utf-8") as f_in, output_file.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                parts = line.strip().split(",")
                if len(parts) >= 14 and "GLN" in parts[4]:
                    try:
                        begin_date_str = parts[6].strip()
                        begin_date = datetime.strptime(begin_date_str, "%Y-%m-%d")

                        end_date_str = parts[13].strip()

                        if begin_date.year <= 2011:
                            if not end_date_str:
                                f_out.write(line)
                            else:
                                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                                if end_date.year >= 2014:
                                    f_out.write(line)

                    except ValueError:
                        continue

    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except Exception as e:
        print(f"An error occurred: {e!s}")


if __name__ == "__main__":
    INTERIM_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    input_file = RAW_WEATHER_DIR / "SRCE.DATA"
    output_file = INTERIM_WEATHER_DIR / "gln_records_2011_2014.txt"

    extract_gln_records(input_file, output_file)
    print(f"Extracted GLN records (2011-2014) have been saved to {output_file}")
