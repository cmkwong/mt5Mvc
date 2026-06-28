from mt5Mvc.models.myUtils.paramModel import DatetimeTuple
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.models.myUtils import timeModel

import config

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os
import re


class Test:
    MainPath = "./docs/pinbar_gambling"
    SampleGridRows = 8
    SampleGridCols = 8
    SampleImagePixels = 2048

    def __init__(self):
        self.mt5Controller = MT5Controller()

    def _detect_pinbar_flags(
        self,
        ohlc: pd.DataFrame,
        body_to_range_max: float = 0.35,
        body_to_range_min: float = 0.05,
        wick_to_body_min: float = 2.0,
        opposite_wick_to_range_max: float = 0.35,
        dominant_tail_to_range_min: float = 0.55,
        body_zone_ratio: float = 0.33,
    ):
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in ohlc.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLC columns: {missing_cols}")

        candles = ohlc.loc[:, required_cols].copy()
        body = (candles["close"] - candles["open"]).abs()
        total_range = (candles["high"] - candles["low"]).replace(0.0, np.nan)
        upper_wick = candles["high"] - candles[["open", "close"]].max(axis=1)
        lower_wick = candles[["open", "close"]].min(axis=1) - candles["low"]
        body_ratio = body / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        body_top = candles[["open", "close"]].max(axis=1)
        body_bottom = candles[["open", "close"]].min(axis=1)

        # Bull pin-bar body should be near candle top; bear pin-bar body near candle bottom.
        bull_body_near_top = body_bottom >= (
            candles["low"] + (1.0 - body_zone_ratio) * total_range
        )
        bear_body_near_bottom = body_top <= (
            candles["low"] + body_zone_ratio * total_range
        )

        common_cond = (
            total_range.notna()
            & (body_ratio <= body_to_range_max)
            & (body_ratio >= body_to_range_min)
        )

        bull_pinbar = (
            common_cond
            & (lower_wick >= wick_to_body_min * body)
            & (lower_wick_ratio >= dominant_tail_to_range_min)
            & (upper_wick <= opposite_wick_to_range_max * total_range)
            & bull_body_near_top
        )
        bear_pinbar = (
            common_cond
            & (upper_wick >= wick_to_body_min * body)
            & (upper_wick_ratio >= dominant_tail_to_range_min)
            & (lower_wick <= opposite_wick_to_range_max * total_range)
            & bear_body_near_bottom
        )

        return bull_pinbar.fillna(False), bear_pinbar.fillna(False)

    def _safe_file_name(self, txt: str):
        return re.sub(r"[^A-Za-z0-9_.-]", "_", txt)

    def _save_params_txt(self, output_dir: str, params: dict):
        params_path = os.path.join(output_dir, "params.txt")
        with open(params_path, "w", encoding="utf-8") as f:
            for key in sorted(params.keys()):
                f.write(f"{key}: {params[key]}\n")
        return params_path

    def _resolve_event_index(self, index: pd.Index, event_time):
        if len(index) == 0:
            return None
        event_ts = pd.Timestamp(event_time)
        if event_ts in index:
            loc = index.get_loc(event_ts)
            if isinstance(loc, slice):
                return int(loc.start)
            if isinstance(loc, np.ndarray):
                true_locs = np.flatnonzero(loc)
                return int(true_locs[0]) if len(true_locs) > 0 else None
            return int(loc)

        pos = int(index.searchsorted(event_ts))
        if pos <= 0:
            return 0
        if pos >= len(index):
            return len(index) - 1

        prev_time = index[pos - 1]
        next_time = index[pos]
        if abs(event_ts - prev_time) <= abs(next_time - event_ts):
            return pos - 1
        return pos

    def _get_forward_points(
        self, close: pd.Series, horizon: int = 10, point_factor: float = 1.0
    ):
        # Summed log-return over the next N bars, then converted to forex points.
        log_ret = np.log(close / close.shift(1))
        future_log_ret = pd.concat(
            [log_ret.shift(-offset) for offset in range(1, horizon + 1)], axis=1
        ).sum(axis=1, min_count=horizon)
        future_return = np.exp(future_log_ret) - 1.0
        return future_return * close * point_factor

    def _get_point_factor(self, symbol: str, all_symbols_info: dict):
        symbol_info = {}
        if isinstance(all_symbols_info, dict):
            symbol_info = all_symbols_info.get(symbol, {})
        digits = symbol_info.get("digits", None)
        if digits is None:
            digits = 3 if symbol.endswith("JPY") else 5
        return 10 ** int(digits)

    def _build_symbol_records(
        self,
        symbol: str,
        ohlc: pd.DataFrame,
        horizon: int,
        point_factor: float,
        body_to_range_max: float,
        body_to_range_min: float,
        wick_to_body_min: float,
        opposite_wick_to_range_max: float,
        dominant_tail_to_range_min: float,
        body_zone_ratio: float,
    ):
        bull_pinbar, bear_pinbar = self._detect_pinbar_flags(
            ohlc,
            body_to_range_max=body_to_range_max,
            body_to_range_min=body_to_range_min,
            wick_to_body_min=wick_to_body_min,
            opposite_wick_to_range_max=opposite_wick_to_range_max,
            dominant_tail_to_range_min=dominant_tail_to_range_min,
            body_zone_ratio=body_zone_ratio,
        )
        forward_points = self._get_forward_points(
            ohlc["close"], horizon=horizon, point_factor=point_factor
        )
        change_col = f"next_{horizon}_bars_points"

        bull_records = pd.DataFrame(
            {
                "symbol": symbol,
                "event_time": forward_points[bull_pinbar].index,
                "signal": "bull",
                change_col: forward_points[bull_pinbar].values,
            }
        )
        bear_records = pd.DataFrame(
            {
                "symbol": symbol,
                "event_time": forward_points[bear_pinbar].index,
                "signal": "bear",
                change_col: forward_points[bear_pinbar].values,
            }
        )

        records = pd.concat([bull_records, bear_records], ignore_index=True)
        records.dropna(subset=[change_col], inplace=True)
        return records

    def _add_stats_overlay(self, ax, values: pd.Series):
        clean_values = pd.Series(values).dropna()
        if len(clean_values) == 0:
            return

        count = len(clean_values)
        mean_value = float(clean_values.mean())
        wins = clean_values[clean_values > 0]
        losses = clean_values[clean_values <= 0]
        win_rate = float(len(wins) / count)
        loss_rate = 1.0 - win_rate
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        e_value = win_rate * avg_win + loss_rate * avg_loss
        percentile_levels = list(range(10, 100, 10))
        percentile_values = np.percentile(clean_values.values, percentile_levels)
        percentile_dict = {
            f"p{level}": float(value)
            for level, value in zip(percentile_levels, percentile_values)
        }

        ax.axvline(
            mean_value,
            color="#1f77b4",
            linestyle="-",
            linewidth=1.2,
            alpha=0.95,
            label="Mean",
        )
        ax.axvline(
            percentile_dict["p10"],
            color="#4d4d4d",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="P10",
        )
        ax.axvline(
            percentile_dict["p50"],
            color="#4d4d4d",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.9,
            label="P50",
        )
        ax.axvline(
            percentile_dict["p90"],
            color="#4d4d4d",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label="P90",
        )

        stats_lines = [
            f"count={count}",
            f"E={e_value:.2f}",
            f"mean={mean_value:.2f}",
            f"win_rate={win_rate * 100:.1f}%",
            f"avg_win={avg_win:.2f}",
            f"avg_loss={avg_loss:.2f}",
        ]
        stats_lines.extend(
            [
                f"p{level}={percentile_dict[f'p{level}']:.2f}"
                for level in percentile_levels
            ]
        )
        stats_text = "\n".join(stats_lines)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )
        ax.legend(loc="upper right", fontsize=8)

    def _plot_symbol_distribution(
        self,
        symbol: str,
        records: pd.DataFrame,
        change_col: str,
        timeframe: str,
        horizon: int,
        period_start: str,
        period_end: str,
        bins: int = 50,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        bull_points = records.loc[records["signal"] == "bull", change_col]
        bear_points = records.loc[records["signal"] == "bear", change_col]
        bull_count = len(bull_points.dropna())
        bear_count = len(bear_points.dropna())
        total_count = bull_count + bear_count
        bull_e = float(bull_points.mean()) if bull_count > 0 else 0.0
        bear_e = float(bear_points.mean()) if bear_count > 0 else 0.0
        total_e = (
            float((bull_points.sum() + bear_points.sum()) / total_count)
            if total_count > 0
            else 0.0
        )

        axes[0].hist(
            bull_points, bins=bins, color="#2ca02c", alpha=0.8, edgecolor="black"
        )
        axes[0].set_title(
            f"Bull Pin-Bar: Next {horizon} Bars Points Distribution (n={bull_count}, E={bull_e:.2f})"
        )
        axes[0].set_xlabel("Points")
        axes[0].set_ylabel("Frequency")
        self._add_stats_overlay(axes[0], bull_points)

        axes[1].hist(
            bear_points, bins=bins, color="#d62728", alpha=0.8, edgecolor="black"
        )
        axes[1].set_title(
            f"Bear Pin-Bar: Next {horizon} Bars Points Distribution (n={bear_count}, E={bear_e:.2f})"
        )
        axes[1].set_xlabel("Points")
        axes[1].set_ylabel("Frequency")
        self._add_stats_overlay(axes[1], bear_points)

        if len(bull_points) == 0:
            axes[0].text(
                0.5,
                0.5,
                "No bull samples",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
        if len(bear_points) == 0:
            axes[1].text(
                0.5,
                0.5,
                "No bear samples",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        fig.suptitle(
            f"{symbol} Pin-Bar Post Signal Distribution ({timeframe}) | period: {period_start} -> {period_end} | total n={total_count}, E={total_e:.2f}",
            fontsize=13,
        )
        return fig

    def _plot_candles(
        self, ax, ohlc_window: pd.DataFrame, event_offset: int, signal: str
    ):
        if ohlc_window.empty:
            ax.text(0.5, 0.5, "No candle data", ha="center", va="center")
            ax.set_axis_off()
            return

        highs = ohlc_window["high"]
        lows = ohlc_window["low"]
        y_span = float(highs.max() - lows.min())
        min_body = y_span * 0.002 if y_span > 0 else 1e-6

        for i, (_, candle) in enumerate(ohlc_window.iterrows()):
            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            color = "#2ca02c" if c >= o else "#d62728"

            ax.vlines(i, l, h, colors=color, linewidth=0.7, alpha=0.95)

            body_y = min(o, c)
            body_h = max(abs(c - o), min_body)
            rect = Rectangle(
                (i - 0.3, body_y),
                0.6,
                body_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.95,
            )
            ax.add_patch(rect)

        ax.axvspan(event_offset - 0.5, event_offset + 0.5, color="#f5c542", alpha=0.25)

        ax.grid(alpha=0.18)
        ax.set_xlim(-1, len(ohlc_window))
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    def _plot_random_samples(
        self,
        records: pd.DataFrame,
        ohlc_by_symbol: dict,
        sample_size: int,
        window_before: int,
        window_after: int,
        random_seed: int,
    ):
        fig_size = self.SampleImagePixels / 100.0
        fig, axes = plt.subplots(
            self.SampleGridRows,
            self.SampleGridCols,
            figsize=(fig_size, fig_size),
            dpi=100,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        if records.empty:
            for ax in axes_flat:
                ax.set_axis_off()
            fig.suptitle(
                f"Random Pin-Bar Samples ({sample_size}): {self.SampleGridRows}x{self.SampleGridCols} Candle Verification | No signals",
                fontsize=16,
            )
            return fig

        sample_count = min(sample_size, len(records), len(axes_flat))
        sampled_records = records.sample(n=sample_count, random_state=random_seed)
        sampled_records = sampled_records.reset_index(drop=True)

        for i, ax in enumerate(axes_flat):
            if i >= sample_count:
                ax.set_axis_off()
                continue

            row = sampled_records.iloc[i]
            symbol = row["symbol"]
            signal = row["signal"]
            event_time = row["event_time"]
            ohlc = ohlc_by_symbol.get(symbol, pd.DataFrame())

            if ohlc.empty:
                ax.text(0.5, 0.5, f"{symbol}\nNo data", ha="center", va="center")
                ax.set_axis_off()
                continue

            event_idx = self._resolve_event_index(ohlc.index, event_time)
            if event_idx is None:
                ax.text(0.5, 0.5, f"{symbol}\nNo match", ha="center", va="center")
                ax.set_axis_off()
                continue

            start_idx = max(0, event_idx - window_before)
            end_idx = min(len(ohlc), event_idx + window_after + 1)
            ohlc_window = ohlc.iloc[start_idx:end_idx]
            event_offset = event_idx - start_idx

            self._plot_candles(ax, ohlc_window, event_offset, signal)
            event_time_txt = pd.Timestamp(event_time).strftime("%Y-%m-%d %H:%M")
            ax.set_title(f"{symbol} | {signal} | {event_time_txt}", fontsize=7)

        fig.suptitle(
            f"Random Pin-Bar Samples ({sample_count}): {self.SampleGridRows}x{self.SampleGridCols} Candle Verification",
            fontsize=16,
        )
        return fig

    def run(
        self,
        *,
        symbols: list = config.Default_Forex_Symbols,
        timeframe: str = "30min",
        start: DatetimeTuple = (2020, 6, 1, 0, 0),
        end: DatetimeTuple = (2025, 7, 30, 23, 59),
        horizon: int = 15,
        bins: int = 75,
        body_to_range_max: float = 0.35,
        body_to_range_min: float = 0.05,
        wick_to_body_min: float = 2.0,
        opposite_wick_to_range_max: float = 0.25,
        dominant_tail_to_range_min: float = 0.55,
        body_zone_ratio: float = 0.33,
        sample_size: int = 64,
        sample_window_before: int = 15,
        sample_window_after: int = 15,
        random_seed: int = 42,
        show_plot: bool = False,
        save_plot: bool = True,
        save_records: bool = True,
        save_distribution: bool = True,
    ):
        run_tag = timeModel.getTimeS(outputFormat="%Y%m%d-%H%M%S")
        os.makedirs(self.MainPath, exist_ok=True)
        output_dir = os.path.join(self.MainPath, run_tag)
        os.makedirs(output_dir, exist_ok=True)

        run_params = {
            "symbols": symbols,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "horizon": horizon,
            "bins": bins,
            "body_to_range_max": body_to_range_max,
            "body_to_range_min": body_to_range_min,
            "wick_to_body_min": wick_to_body_min,
            "opposite_wick_to_range_max": opposite_wick_to_range_max,
            "dominant_tail_to_range_min": dominant_tail_to_range_min,
            "body_zone_ratio": body_zone_ratio,
            "sample_size": sample_size,
            "sample_window_before": sample_window_before,
            "sample_window_after": sample_window_after,
            "random_seed": random_seed,
            "result_metric": "points",
            "show_plot": show_plot,
            "save_plot": save_plot,
            "save_records": save_records,
            "save_distribution": save_distribution,
        }
        params_path = self._save_params_txt(output_dir, run_params)
        period_start = timeModel.getTimeS(start, "%Y-%m-%d %H:%M")
        period_end = timeModel.getTimeS(end, "%Y-%m-%d %H:%M")

        prices = self.mt5Controller.pricesLoader.getPrices(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            ohlcvs="111100",
        )

        ohlc_by_symbol = prices.get_ohlcvs_from_prices()
        all_symbols_info = getattr(prices, "all_symbols_info", {})
        records_list = []
        records_by_symbol = {}
        for symbol in symbols:
            if symbol not in ohlc_by_symbol:
                continue
            point_factor = self._get_point_factor(symbol, all_symbols_info)
            symbol_records = self._build_symbol_records(
                symbol,
                ohlc_by_symbol[symbol],
                horizon=horizon,
                point_factor=point_factor,
                body_to_range_max=body_to_range_max,
                body_to_range_min=body_to_range_min,
                wick_to_body_min=wick_to_body_min,
                opposite_wick_to_range_max=opposite_wick_to_range_max,
                dominant_tail_to_range_min=dominant_tail_to_range_min,
                body_zone_ratio=body_zone_ratio,
            )
            records_list.append(symbol_records)
            records_by_symbol[symbol] = symbol_records

        change_col = f"next_{horizon}_bars_points"
        if records_list:
            records = pd.concat(records_list, ignore_index=True)
            records.sort_values(by=["symbol", "event_time", "signal"], inplace=True)
            records.reset_index(drop=True, inplace=True)
        else:
            records = pd.DataFrame(
                columns=["symbol", "event_time", "signal", change_col]
            )

        csv_path = None
        if save_records:
            csv_path = os.path.join(output_dir, "pinbar_records.csv")
            records.to_csv(csv_path, index=False)

        symbol_figure_paths = {}
        if save_plot and save_distribution:
            for symbol in symbols:
                symbol_records = records_by_symbol.get(
                    symbol,
                    pd.DataFrame(
                        columns=["symbol", "event_time", "signal", change_col]
                    ),
                )
                fig = self._plot_symbol_distribution(
                    symbol=symbol,
                    records=symbol_records,
                    change_col=change_col,
                    timeframe=timeframe,
                    horizon=horizon,
                    period_start=period_start,
                    period_end=period_end,
                    bins=bins,
                )
                fig_path = os.path.join(
                    output_dir,
                    f"{self._safe_file_name(symbol)}_distribution.png",
                )
                fig.savefig(fig_path, dpi=150)
                symbol_figure_paths[symbol] = fig_path
                plt.close(fig)

        sample_figure = self._plot_random_samples(
            records=records,
            ohlc_by_symbol=ohlc_by_symbol,
            sample_size=sample_size,
            window_before=sample_window_before,
            window_after=sample_window_after,
            random_seed=random_seed,
        )
        sample_figure_path = None
        if save_plot:
            sample_figure_path = os.path.join(
                output_dir, f"pinbar_random_sample_{sample_size}.png"
            )
            sample_figure.savefig(sample_figure_path, dpi=100)

        if show_plot:
            plt.show()
        plt.close(sample_figure)

        return {
            "records": records,
            "output_dir": output_dir,
            "params_path": params_path,
            "symbol_figure_paths": symbol_figure_paths,
            "sample_figure_path": sample_figure_path,
            "csv_path": csv_path,
            "figure_path": sample_figure_path,
        }

    def run_sample_only(
        self,
        *,
        symbol: str = "USDJPY",
        timeframe: str = "30min",
        start: DatetimeTuple = (2020, 6, 1, 0, 0),
        end: DatetimeTuple = (2025, 7, 30, 23, 59),
        horizon: int = 15,
        body_to_range_max: float = 0.35,
        body_to_range_min: float = 0.05,
        wick_to_body_min: float = 2.0,
        opposite_wick_to_range_max: float = 0.35,
        dominant_tail_to_range_min: float = 0.55,
        body_zone_ratio: float = 0.33,
        sample_size: int = 64,
        sample_window_before: int = 15,
        sample_window_after: int = 15,
        random_seed: int = 42,
        show_plot: bool = False,
        save_plot: bool = True,
        save_records: bool = True,
    ):
        return self.run(
            symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
            horizon=horizon,
            body_to_range_max=body_to_range_max,
            body_to_range_min=body_to_range_min,
            wick_to_body_min=wick_to_body_min,
            opposite_wick_to_range_max=opposite_wick_to_range_max,
            dominant_tail_to_range_min=dominant_tail_to_range_min,
            body_zone_ratio=body_zone_ratio,
            sample_size=sample_size,
            sample_window_before=sample_window_before,
            sample_window_after=sample_window_after,
            random_seed=random_seed,
            show_plot=show_plot,
            save_plot=save_plot,
            save_records=save_records,
            save_distribution=False,
        )
