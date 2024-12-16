import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    @staticmethod
    def evaluate_72h_predictions(
        actual: np.ndarray, predicted: np.ndarray, window_size: int = 144, step_size: int = 144
    ) -> list[dict]:
        """Evaluate 72-hour predictions.

        Args:
            actual: Actual values
            predicted: Predicted values
            window_size: Size of prediction window (default: 144 for 72 hours)
            step_size: Step size between windows (default: 144)

        Returns:
            List of dictionaries containing evaluation metrics for each window
        """
        evaluation_results = []

        for i in range(0, len(actual) - window_size + 1, step_size):
            window_actual = actual[i : i + window_size]
            window_pred = predicted[i : i + window_size]

            metrics = {
                "window_start_idx": i,
                "MAE_24h": mean_absolute_error(window_actual[:48], window_pred[:48]),
                "RMSE_24h": np.sqrt(mean_squared_error(window_actual[:48], window_pred[:48])),
                "MAE_48h": mean_absolute_error(window_actual[48:96], window_pred[48:96]),
                "RMSE_48h": np.sqrt(mean_squared_error(window_actual[48:96], window_pred[48:96])),
                "MAE_72h": mean_absolute_error(window_actual[96:], window_pred[96:]),
                "RMSE_72h": np.sqrt(mean_squared_error(window_actual[96:], window_pred[96:])),
            }

            evaluation_results.append(metrics)

        return evaluation_results

    @staticmethod
    def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Evaluate predictions.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary containing evaluation metrics
        """
        return {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100,
        }
