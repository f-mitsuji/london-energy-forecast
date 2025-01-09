import logging
from typing import Any

import numpy as np
import xgboost as xgb

from src.ml.base_model import BaseModel
from src.ml.data_preparation import DataPreparation
from src.ml.model_evaluator import ModelEvaluator
from src.ml.visualizer import Visualizer
from src.utils import setup_logger


class XGBoostModel(BaseModel):
    def __init__(self, **params: Any):
        super().__init__()
        self.default_params = {
            "objective": "reg:squarederror",
            "validate_parameters": True,
            "random_state": 1,
            "n_estimators": 100,
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

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            msg = "Model must be trained before making predictions"
            raise ValueError(msg)

        return self.model.predict(X)

    def predict_72h(
        self, test_df, train_df, y_test, target_col: str = "demand", debug_output_path: str = "debug_output"
    ):
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        # デバッグ出力用のディレクトリ作成
        import os

        os.makedirs(debug_output_path, exist_ok=True)

        # 基本設定
        window_size = 144  # 72時間分
        n_windows = len(test_df) // window_size
        predictions = []

        # データをnumpy配列に変換
        y_train = train_df.to_numpy()
        y_test = y_test.to_numpy()

        logger = logging.getLogger("xgboost")
        logger.info(f"Starting predictions for {n_windows} windows")

        for window_idx in range(n_windows):
            # デバッグ情報を格納するリスト
            debug_info = []

            # ウィンドウの範囲を設定
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size
            window_data = test_df.iloc[start_idx:end_idx].copy()
            window_actual = y_test[start_idx:end_idx]

            logger.info(f"Processing window {window_idx + 1}/{n_windows}")

            # このウィンドウでの予測
            for i in range(len(window_data)):
                point_in_cycle = i
                hours = point_in_cycle / 2.0

                # 現在の特徴量でのlag値を記録
                current_lags = {col: window_data.iloc[i][col] for col in window_data.columns if "lag" in col}

                # 予測
                current_features = window_data.iloc[[i]]
                pred = float(self.predict(current_features)[0])
                predictions.append(pred)

                # デバッグ情報を収集
                debug_row = {
                    "window": window_idx + 1,
                    "point": point_in_cycle,
                    "hours": hours,
                    "prediction": pred,
                    "actual": window_actual[i],
                }
                debug_row.update(current_lags)
                debug_info.append(debug_row)

                # 次のステップのlag特徴量を更新
                if i < len(window_data) - 1:
                    next_idx = window_data.index[i + 1]
                    for col in window_data.columns:
                        if "lag" in col:
                            hours = float(col.split("_")[-1].replace("h", ""))
                            points = int(hours * 2)

                            # 全体のインデックスを計算
                            global_idx = len(y_train) + start_idx + i + 1
                            history_idx = global_idx - points

                            # テストデータのインデックスを計算
                            test_data_index = history_idx - len(y_train)

                            if history_idx >= len(y_train) and test_data_index < start_idx:
                                # テストデータの範囲内かつ現在のウィンドウより前のデータ
                                window_data.loc[next_idx, col] = y_test[test_data_index]
                            elif history_idx >= len(y_train):
                                # 現在のウィンドウ内の予測値を使用
                                window_data.loc[next_idx, col] = predictions[test_data_index]
                            else:
                                # 学習データの範囲内
                                window_data.loc[next_idx, col] = y_train[history_idx]

                # 進捗の出力
                if point_in_cycle == 0:
                    print(f"\n=== Window {window_idx + 1}: 0時間経過（開始） ===")
                elif point_in_cycle == 48:
                    print("\n=== 24時間経過 ===")
                elif point_in_cycle == 96:
                    print("\n=== 48時間経過 ===")
                elif point_in_cycle == 143:
                    print("\n=== 72時間経過（ウィンドウ終了） ===")
                    print("-" * 50)

            # デバッグ情報をCSVファイルに出力
            import pandas as pd

            debug_df = pd.DataFrame(debug_info)
            output_file = os.path.join(debug_output_path, f"debug_window_{window_idx + 1}.csv")
            debug_df.to_csv(output_file, index=False)
            logger.info(f"Saved debug information for window {window_idx + 1}")

        return np.array(predictions)

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
    logger = setup_logger("xgboost")
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
    # data_prep = DataPreparation("MAC000145_ml_ready.csv", *features_to_drop)
    data_prep = DataPreparation("MAC000152_ml_ready.csv", *features_to_drop)
    # data_prep = DataPreparation("MAC000002_ml_ready.csv", *features_to_drop)

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

    logger.info("Starting 72-hour predictions")
    predictions = model.predict_72h(X_test, y_train, y_test)
    logger.info("Predictions completed")

    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_72h_predictions(actual=y_test.to_numpy(), predicted=predictions)

    # 平均RMSEの計算と表示
    logger.info("\nAverage RMSE by Horizon:")
    avg_rmse = {
        horizon: np.mean([result[f"RMSE_{horizon}"] for result in evaluation_results])
        for horizon in ["24h", "48h", "72h"]
    }
    for horizon, rmse in avg_rmse.items():
        logger.info(f"{horizon}: {rmse:.4f}")

    # 全体のRMSEを表示（予測値の長さに合わせる）
    valid_length = len(predictions)
    overall_metrics = evaluator.evaluate_predictions(y_test[:valid_length], predictions)
    logger.info(f"\nOverall RMSE: {overall_metrics['RMSE']:.4f}")

    feature_importance = model.get_feature_importance(feature_names=X_train.columns.tolist())
    print("\nFeature Importance:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")

    visualizer = Visualizer("XGBoost")
    visualizer.plot_predictions(y_test, predictions)
    visualizer.plot_prediction_windows(y_test, predictions)
    visualizer.plot_metrics(evaluation_results)
    logger.info("Visualization completed")


if __name__ == "__main__":
    main()
