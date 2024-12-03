from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

RAW_ENERGY_DIR = RAW_DIR / "energy"
RAW_WEATHER_DIR = RAW_DIR / "weather"

INTERIM_ENERGY_DIR = INTERIM_DIR / "energy"
INTERIM_WEATHER_DIR = INTERIM_DIR / "weather"

PROCESSED_ENERGY_DIR = PROCESSED_DIR / "energy"
PROCESSED_WEATHER_DIR = PROCESSED_DIR / "weather"

REPORTS_DIR = BASE_DIR / "reports"
WEATHER_REPORTS_DIR = REPORTS_DIR / "weather"
MODELS_REPORTS_DIR = REPORTS_DIR / "models"

MODELS_DIR = BASE_DIR / "models"
