import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.settings import PROCESSED_DIR


class DataPreparation:
    def __init__(self, file_name: str, *features_to_drop: str):
        self.file_path = PROCESSED_DIR / file_name
        self.default_features_to_drop = [
            "timestamp",
            "cooling_degree_squared",
            "heating_degree_squared",
            "discomfort_index",
            "sun_duration",
            # "cloud_cover",
        ]
        self.features_to_drop = [*self.default_features_to_drop, *features_to_drop]

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        return df.drop(self.features_to_drop, axis=1).dropna()

    def get_feature_target_split(self, target_col: str = "demand") -> tuple[pd.DataFrame, pd.Series]:
        df = self.load_data()
        return df.drop(target_col, axis=1), df[target_col]

    def get_train_test_split(self, test_size: float = 0.2, valid_size: float | None = None) -> dict:
        features, target = self.get_feature_target_split()

        if valid_size is None:
            test_idx = int(len(features) * (1 - test_size))

            return {
                "X_train": features[:test_idx],
                "X_test": features[test_idx:],
                "y_train": target[:test_idx],
                "y_test": target[test_idx:],
            }

        test_idx = int(len(features) * (1 - test_size))
        valid_idx = int(test_idx * (1 - valid_size))

        return {
            "X_train": features[:valid_idx],
            "X_valid": features[valid_idx:test_idx],
            "X_test": features[test_idx:],
            "y_train": target[:valid_idx],
            "y_valid": target[valid_idx:test_idx],
            "y_test": target[test_idx:],
        }

    def get_sequence_split(self, df: pd.DataFrame, seq_length: int = 48, test_size: float = 0.2) -> tuple:
        features, target = self.get_feature_target_split()

        train_size = int(len(df) * (1 - test_size))
        train_features = features[:train_size]
        test_features = features[train_size:]
        train_target = target[:train_size]
        test_target = target[train_size:]

        scale_cols = ["cooling_degree", "heating_degree", "discomfort_index"]
        feature_scaler = MinMaxScaler()
        train_features[scale_cols] = feature_scaler.fit_transform(train_features[scale_cols])
        test_features[scale_cols] = feature_scaler.transform(test_features[scale_cols])

        train_features_array = train_features.to_numpy()
        test_features_array = test_features.to_numpy()

        target_scaler = MinMaxScaler()
        train_target_scaled = target_scaler.fit_transform(train_target.to_numpy().reshape(-1, 1)).ravel()
        test_target_scaled = target_scaler.transform(test_target.to_numpy().reshape(-1, 1)).ravel()

        X_train = [train_features_array[i : i + seq_length] for i in range(len(train_features_array) - seq_length)]
        y_train = train_target_scaled[seq_length:]

        X_test = [test_features_array[i : i + seq_length] for i in range(len(test_features_array) - seq_length)]
        y_test = test_target_scaled[seq_length:]

        return (
            np.array(X_train),
            np.array(X_test),
            y_train,
            y_test,
            feature_scaler,
            target_scaler,
        )
