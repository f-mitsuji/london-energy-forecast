import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    @staticmethod
    def evaluate_72h_predictions(
        actual: np.ndarray, predicted: np.ndarray, window_size: int = 144, step_size: int = 144
    ) -> list[dict]:
        """72時間予測の評価を行う

        Args:
            actual: 実測値の配列
            predicted: 予測値の配列
            window_size: 予測ウィンドウのサイズ（デフォルト: 72時間 = 144ポイント）
            step_size: ウィンドウ間のステップサイズ（デフォルト: 144）

        Returns:
            各ウィンドウの評価指標を含む辞書のリスト
        """
        evaluation_results = []

        for i in range(0, len(actual) - window_size + 1, step_size):
            window_actual = actual[i : i + window_size]
            window_pred = predicted[i : i + window_size]

            # 24時間 (0-48ポイント)
            points_24h = 48
            actual_24h = window_actual[:points_24h]
            pred_24h = window_pred[:points_24h]

            # 48時間 (0-96ポイント)
            points_48h = 96
            actual_48h = window_actual[:points_48h]
            pred_48h = window_pred[:points_48h]

            # 72時間 (0-144ポイント)
            actual_72h = window_actual
            pred_72h = window_pred

            metrics = {
                "window_start_idx": i,
                # 24時間の評価
                "MAE_24h": mean_absolute_error(actual_24h, pred_24h),
                "RMSE_24h": np.sqrt(mean_squared_error(actual_24h, pred_24h)),
                "MAPE_24h": np.mean(np.abs((actual_24h - pred_24h) / actual_24h)) * 100,
                # 48時間の評価
                "MAE_48h": mean_absolute_error(actual_48h, pred_48h),
                "RMSE_48h": np.sqrt(mean_squared_error(actual_48h, pred_48h)),
                "MAPE_48h": np.mean(np.abs((actual_48h - pred_48h) / actual_48h)) * 100,
                # 72時間の評価
                "MAE_72h": mean_absolute_error(actual_72h, pred_72h),
                "RMSE_72h": np.sqrt(mean_squared_error(actual_72h, pred_72h)),
                "MAPE_72h": np.mean(np.abs((actual_72h - pred_72h) / actual_72h)) * 100,
            }

            # 時間帯ごとの評価も追加（オプション）
            if False:  # 必要な場合はTrueに変更
                # 24-48時間の評価
                actual_24_48h = window_actual[points_24h:points_48h]
                pred_24_48h = window_pred[points_24h:points_48h]
                metrics.update(
                    {
                        "MAE_24_48h": mean_absolute_error(actual_24_48h, pred_24_48h),
                        "RMSE_24_48h": np.sqrt(mean_squared_error(actual_24_48h, pred_24_48h)),
                        "MAPE_24_48h": np.mean(np.abs((actual_24_48h - pred_24_48h) / actual_24_48h)) * 100,
                    }
                )

                # 48-72時間の評価
                actual_48_72h = window_actual[points_48h:]
                pred_48_72h = window_pred[points_48h:]
                metrics.update(
                    {
                        "MAE_48_72h": mean_absolute_error(actual_48_72h, pred_48_72h),
                        "RMSE_48_72h": np.sqrt(mean_squared_error(actual_48_72h, pred_48_72h)),
                        "MAPE_48_72h": np.mean(np.abs((actual_48_72h - pred_48_72h) / actual_48_72h)) * 100,
                    }
                )

            evaluation_results.append(metrics)

        return evaluation_results

    @staticmethod
    def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
        """全体の予測を評価

        Args:
            actual: 実測値の配列
            predicted: 予測値の配列

        Returns:
            評価指標を含む辞書
        """
        # ゼロ除算を避けるための処理を追加
        mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, np.inf, actual))) * 100

        return {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAPE": mape,
        }
