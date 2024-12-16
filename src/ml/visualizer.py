from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.settings import MODELS_REPORTS_DIR


class Visualizer:
    def __init__(self, save_dir: str = "figures"):
        self.save_dir = MODELS_REPORTS_DIR / save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Demand Prediction",
        *,
        save: bool = True,
        show: bool = True,
        timestamp: np.ndarray | None = None,
    ) -> None:
        plt.figure(figsize=(15, 6))

        # 予測値の長さに合わせて実測値を切り詰める
        actual = actual[: len(predicted)]

        # 時系列のインデックスを設定
        x = timestamp[: len(predicted)] if timestamp is not None else np.arange(len(predicted))

        # 実測値と予測値をプロット
        plt.plot(x, actual, label="Actual", color="blue", alpha=0.5)
        plt.plot(x, predicted, label="Predicted", color="red", alpha=0.5)

        plt.title(title)
        plt.xlabel("Time" if timestamp is not None else "Index")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(visible=True, alpha=0.3)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    def plot_prediction_windows(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        window_size: int = 144,
        step_size: int = 144,
        max_windows: int | None = None,
        *,
        save: bool = True,
        show: bool = True,
    ) -> None:
        num_windows = (len(actual) - window_size + 1) // step_size
        if max_windows is not None:
            num_windows = min(num_windows, max_windows)

        fig, axes = plt.subplots(num_windows, 1, figsize=(15, 5 * num_windows))
        if num_windows == 1:
            axes = [axes]

        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            # 24時間ごとの境界線のx座標
            boundaries = [start_idx + j * 48 for j in range(4)]

            ax = axes[i]
            ax.plot(actual[start_idx:end_idx], label="Actual", color="blue", alpha=0.5)
            ax.plot(predicted[start_idx:end_idx], label="Predicted", color="red", alpha=0.5)

            # 24時間ごとの境界線を描画
            for b in boundaries:
                ax.axvline(x=b - start_idx, color="gray", linestyle="--", alpha=0.3)

            ax.set_title(f"Window {i+1}")
            ax.set_xlabel("Steps (30-min intervals)")
            ax.set_ylabel("Demand")
            ax.legend()
            ax.grid(visible=True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_windows_{timestamp}.png"
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    def plot_metrics(self, evaluation_results: list[dict], *, save: bool = True, show: bool = True) -> None:
        metrics = ["RMSE_24h", "RMSE_48h", "RMSE_72h"]
        values = {metric: [] for metric in metrics}

        for result in evaluation_results:
            for metric in metrics:
                values[metric].append(result[metric])

        plt.figure(figsize=(10, 6))

        for metric in metrics:
            plt.plot(values[metric], label=metric, marker="o")

        plt.title("Prediction Error by Window")
        plt.xlabel("Window Index")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(visible=True, alpha=0.3)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_metrics_{timestamp}.png"
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()
