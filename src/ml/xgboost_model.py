import logging
import sys
from typing import Any

import numpy as np
import xgboost as xgb

from src.ml.base_model import BaseModel
from src.ml.data_preparation import DataPreparation
from src.ml.model_evaluator import ModelEvaluator
from src.utils import setup_logger


class XGBoostModel(BaseModel):
    def __init__(self, **params: Any):
        super().__init__()
        self.default_params = {
            "objective": "reg:squarederror",
            "validate_parameters": True,
            "random_state": 1,
            # "device": "cuda",
            "n_estimators": 1000,
            "tree_method": "hist",
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self.model_params = {**self.default_params, **params}
        self.model: xgb.XGBRegressor = None

    def train(self, X, y, **kwargs) -> None:
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

    def _extract_period(self, time_str: str) -> int:
        numeric_part = time_str.replace("h", "")
        hours = float(numeric_part)
        return int(hours * 2)

    def predict(self, X, decimals: int = 3) -> np.ndarray:
        if not self.is_fitted:
            msg = "Model must be trained before making predictions"
            raise ValueError(msg)

        predictions = self.model.predict(X)

        # 予測値を丸める
        # rounded_predictions = np.round(predictions, decimals)

        return predictions

    def predict_72h(self, test_df, train_df, target_col: str = "demand"):
        logger = logging.getLogger("energy_forecast")

        if not self.is_fitted:
            msg = "Model must be trained before making predictions"
            logger.error(msg)
            sys.exit(1)

        # Initialize data
        original_df = test_df.copy()
        full_history = list(train_df[target_col].values)
        final_predictions = []
        all_predictions = []
        feature_cols = test_df.columns.tolist()

        # Extract lag features
        lag_cols = [col for col in feature_cols if "lag" in col]
        lag_periods = [self._extract_period(col.split("_")[-1]) for col in lag_cols]
        logger.debug(f"Lag features: {list(zip(lag_cols, lag_periods, strict=True))}")

        # Extract rolling features and build patterns
        rolling_cols = [col for col in feature_cols if "rolling" in col]
        rolling_info = []
        for col in rolling_cols:
            parts = col.split("_")
            hours = float(parts[-2].replace("h", ""))
            stat = parts[-1]
            points = int(hours * 2)  # 時間を30分単位のポイント数に変換
            rolling_info.append((stat, points))

        # 統計量ごとにwindowサイズをグループ化
        rolling_patterns = {}
        for stat, points in rolling_info:
            if stat not in rolling_patterns:
                rolling_patterns[stat] = set()
            rolling_patterns[stat].add(points)
        rolling_patterns = {stat: sorted(list(points)) for stat, points in rolling_patterns.items()}

        logger.debug("Rolling patterns:")
        for stat, points in rolling_patterns.items():
            hours = [p / 2 for p in points]
            logger.debug(f"  {stat}: {points} points ({hours} hours)")

        # Prediction cycle (72 hours = 144 points)
        cycle_length = 144

        for i in range(len(test_df)):
            point_in_cycle = i % cycle_length
            hours = point_in_cycle / 2.0

            # Make prediction
            current_features = test_df.iloc[[i]]
            pred = self.predict(current_features)[0]
            all_predictions.append(pred)
            final_predictions.append(pred)
            full_history.append(pred)

            # Update features for next prediction
            if i < len(test_df) - 1:
                next_row_idx = test_df.iloc[[i + 1]].index[0]

                # Update lag features
                for lag, col in zip(lag_periods, lag_cols, strict=True):
                    if i + 1 >= lag:
                        history_idx = len(full_history) - lag
                        if history_idx >= 0:
                            test_df.loc[next_row_idx, col] = full_history[history_idx]

                # Update rolling features
                current_full_history = full_history[: idx + 1]
                for stat, points_list in rolling_patterns.items():
                    for points in points_list:
                        hours = points // 2
                        col = f"demand_rolling_{hours}h_{stat}"

                        if col not in test_df.columns:
                            continue

                        # Get data for rolling window (closed='left')
                        end_idx = i + 1
                        start_idx = max(0, end_idx - points)
                        window_data = current_full_history[start_idx:end_idx]

                        if len(window_data) == 0:
                            continue

                        # Calculate rolling statistics
                        if stat == "mean":
                            new_value = np.mean(window_data)
                        elif stat == "std":
                            new_value = np.std(window_data) if len(window_data) > 1 else 0
                        elif stat == "min":
                            new_value = np.min(window_data)
                        elif stat == "max":
                            new_value = np.max(window_data)

                        test_df.loc[next_row_idx, col] = new_value

            # Log progress
            if point_in_cycle == 0:
                logger.info(f"Completed {i//2} hours of predictions")
                print("\n=== 0時間経過（新サイクル開始） ===")
            elif point_in_cycle == 48:
                print("\n=== 24時間経過 ===")
            elif point_in_cycle == 96:
                print("\n=== 48時間経過 ===")
            elif point_in_cycle == 143:
                print("\n=== 72時間経過（サイクル終了） ===")
                print("-" * 50)

            print(f"i: {i:3d} | cycle point: {point_in_cycle:3d} | hours: {hours:5.1f}")

            # Reset cycle if needed
            if point_in_cycle == 143 and i + 1 < len(test_df):
                start_idx = (i + 1) - 143
                end_idx = i + 1
                test_df.iloc[start_idx : end_idx + 1] = original_df.iloc[start_idx : end_idx + 1]
                all_predictions = []

                # Update full history with actual values
                full_history[:] = (
                    list(train_df[target_col].values)
                    + list(original_df[target_col].values[:start_idx])
                    + list(final_predictions[start_idx:])
                )

        return np.array(final_predictions)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        if not self.is_fitted:
            msg = "Model must be trained before getting feature importance"
            raise ValueError(msg)

        importance_scores = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]

        importance_dict = dict(zip(feature_names, importance_scores, strict=False))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def main():
    logger = setup_logger()
    logger.info("Starting model training and prediction")

    model = XGBoostModel()
    logger.debug(f"Model parameters: {model.model_params}")

    features_to_drop = [
        "demand_rolling_2h_mean",
        "demand_rolling_4h_mean",
        "demand_rolling_6h_mean",
        "demand_rolling_2h_std",
        "demand_rolling_4h_std",
        "demand_rolling_2h_min",
        "demand_rolling_2h_max",
    ]
    data_prep = DataPreparation("MAC000145_ml_ready.csv", *features_to_drop)

    data = data_prep.get_train_test_split(test_size=0.2)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    logger.info("Starting model training")
    model.train(
        X_train,
        y_train,
        verbose=2,
    )
    logger.info("Model training completed")

    # logger.info("Starting 72-hour predictions")
    # predictions = model.predict_72h(X_test, y_train)
    # logger.info("Predictions completed")

    predictions = model.predict(X_test)

    evaluator = ModelEvaluator()
    # evaluation_results = evaluator.evaluate_72h_predictions(actual=y_test.to_numpy(), predicted=predictions)
    evaluation_results = evaluator.evaluate_predictions(actual=y_test.to_numpy(), predicted=predictions)

    # for i, result in enumerate(evaluation_results):
    #     logger.info(f"\nWindow {i + 1}:")
    #     logger.info(f"24h RMSE: {result['RMSE_24h']:.4f}")
    #     logger.info(f"48h RMSE: {result['RMSE_48h']:.4f}")
    #     logger.info(f"72h RMSE: {result['RMSE_72h']:.4f}")

    for key, value in evaluation_results.items():
        logger.info(f"{key}: {value:.4f}")

    # feature_importance = model.get_feature_importance(feature_names=X_train.columns.tolist())
    # print("\nFeature Importance:")
    # for feature, importance in feature_importance.items():
    #     print(f"{feature}: {importance:.4f}")

    # visualizer = Visualizer("XGBoost")

    # visualizer.plot_predictions(
    #     actual=y_test.to_numpy(), predicted=predictions, title=f"Demand Prediction ({len(predictions)} points)"
    # )

    # visualizer.plot_prediction_windows(
    #     actual=y_test.to_numpy(),
    #     predicted=predictions,
    #     max_windows=3,
    # )

    # visualizer.plot_metrics(evaluation_results)
    # logger.info("Visualization completed")


if __name__ == "__main__":
    main()
