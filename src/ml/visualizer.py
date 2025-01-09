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

        # 予測誤差を計算して表示
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        plt.title(f"{title}\nMAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

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
        n_windows: int = 3,
        *,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """予測ウィンドウをプロット

        Args:
            actual: 実測値
            predicted: 予測値
            window_size: ウィンドウサイズ（デフォルト: 144ポイント = 72時間）
            step_size: ステップサイズ（デフォルト: 144）
            n_windows: 表示するウィンドウ数（デフォルト: 3）
        """
        # 利用可能なウィンドウ数を計算
        total_windows = (len(actual) - window_size + 1) // step_size
        n_windows = min(n_windows, total_windows)

        # フィギュアサイズを調整
        fig, axes = plt.subplots(n_windows, 1, figsize=(15, 4 * n_windows))
        if n_windows == 1:
            axes = [axes]

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            window_actual = actual[start_idx:end_idx]
            window_pred = predicted[start_idx:end_idx]

            # x軸のポイントを生成
            x = np.arange(window_size)

            ax = axes[i]
            ax.plot(x, window_actual, label="Actual", color="blue", alpha=0.5)
            ax.plot(x, window_pred, label="Predicted", color="red", alpha=0.5)

            # 24時間ごとの境界線
            for hour, label in [(0, "0h"), (48, "24h"), (96, "48h"), (144, "72h")]:
                ax.axvline(x=hour, color="gray", linestyle="--", alpha=0.3)
                if hour < 144:  # 最後の72hラベルは省略
                    ax.text(hour, ax.get_ylim()[1], label, horizontalalignment="center", verticalalignment="bottom")

            # 各区間のMAPEを計算して表示
            for start, end, label in [(0, 48, "0-24h"), (48, 96, "24-48h"), (96, 144, "48-72h")]:
                section_actual = window_actual[start:end]
                section_pred = window_pred[start:end]
                mape = np.mean(np.abs((section_actual - section_pred) / section_actual)) * 100
                rmse = np.sqrt(np.mean((section_actual - section_pred) ** 2))

                # テキストの位置を調整
                mid_point = (start + end) // 2
                ax.text(
                    mid_point,
                    ax.get_ylim()[1] * 0.9,
                    f"{label}\nMAPE: {mape:.1f}%\nRMSE: {rmse:.3f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            ax.set_title(f"Window {i+1}")
            ax.set_xlabel("Steps (30-min intervals)")
            ax.set_ylabel("Demand")
            ax.set_xlim(0, window_size)  # x軸の範囲を明示的に設定
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_windows_{timestamp}.png"
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    def plot_metrics(self, evaluation_results: list[dict], *, save: bool = True, show: bool = True) -> None:
        # 評価指標の設定
        metrics = ["RMSE_24h", "RMSE_48h", "RMSE_72h", "MAPE_24h", "MAPE_48h", "MAPE_72h"]
        metric_groups = {"RMSE": ["RMSE_24h", "RMSE_48h", "RMSE_72h"], "MAPE": ["MAPE_24h", "MAPE_48h", "MAPE_72h"]}

        # サブプロットの作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # RMSEのプロット
        for metric in metric_groups["RMSE"]:
            values = [result[metric] for result in evaluation_results]
            ax1.plot(values, label=metric, marker="o")

        ax1.set_title("RMSE by Window")
        ax1.set_xlabel("Window Index")
        ax1.set_ylabel("RMSE")
        ax1.legend()
        ax1.grid(visible=True, alpha=0.3)

        # MAPEのプロット
        for metric in metric_groups["MAPE"]:
            values = [result[metric] for result in evaluation_results]
            ax2.plot(values, label=metric, marker="o")

        ax2.set_title("MAPE by Window")
        ax2.set_xlabel("Window Index")
        ax2.set_ylabel("MAPE (%)")
        ax2.legend()
        ax2.grid(visible=True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_metrics_{timestamp}.png"
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()
