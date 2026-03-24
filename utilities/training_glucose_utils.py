from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


@dataclass
class AnalysisConfig:
    intervals_base_url: str
    intervals_api_key: str
    intervals_athlete_id: str = "0"
    intervals_streams_base_url: str = ""
    nightscout_base_url: str = ""
    nightscout_token: str = ""
    nightscout_api_secret: str = ""
    pre_activity_hours: int = 2
    post_activity_hours: int = 6
    merge_tolerance: str = "3min"


def intervals_get_basic(base_url: str, api_key: str, path: str, params: Optional[dict] = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(url, params=params or {}, auth=("API_KEY", api_key), timeout=30)
    response.raise_for_status()
    return response.json()


def nightscout_auth_headers(token: str = "", api_secret: str = "") -> Dict[str, str]:
    if token:
        return {}
    if api_secret:
        hashed = hashlib.sha1(api_secret.encode("utf-8")).hexdigest()
        return {"API-SECRET": hashed}
    return {}


def nightscout_get(base_url: str, path: str, params: Optional[dict] = None, token: str = "", api_secret: str = "") -> dict:
    params = dict(params or {})
    if token:
        params["token"] = token

    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(url, params=params, headers=nightscout_auth_headers(token=token, api_secret=api_secret), timeout=30)
    response.raise_for_status()
    return response.json()


def list_activities(cfg: AnalysisConfig, days_back: int = 90) -> pd.DataFrame:
    today = pd.Timestamp.now().date()
    oldest = (today - pd.Timedelta(days=days_back)).isoformat()
    newest = (today + pd.Timedelta(days=1)).isoformat()

    activities = intervals_get_basic(
        cfg.intervals_base_url,
        cfg.intervals_api_key,
        f"/api/v1/athlete/{cfg.intervals_athlete_id}/activities",
        params={"oldest": oldest, "newest": newest},
    )

    if not activities:
        raise ValueError("No activities returned by Intervals.icu for selected window.")

    df = pd.DataFrame(activities)
    if "start_date_local" in df.columns:
        df["start_date_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")
        df = df.sort_values("start_date_local", ascending=False)

    show_cols = [c for c in ["id", "start_date_local", "type", "name", "distance", "moving_time", "elapsed_time"] if c in df.columns]
    return df[show_cols + [c for c in df.columns if c not in show_cols]]


def choose_activity(df: pd.DataFrame, index: int) -> dict:
    if index < 0 or index >= len(df):
        raise IndexError(f"Activity index {index} out of range (0-{len(df)-1}).")
    return df.iloc[index].to_dict()


def get_activity_window(activity: dict, pre_hours: int = 2, post_hours: int = 6) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(activity["start_date_local"])
    duration_seconds = activity.get("elapsed_time") or activity.get("moving_time") or 3600
    end = start + pd.to_timedelta(int(duration_seconds), unit="s")
    window_start = start - pd.to_timedelta(pre_hours, unit="h")
    window_end = end + pd.to_timedelta(post_hours, unit="h")
    return start, end, window_start, window_end


def fetch_glucose(cfg: AnalysisConfig, window_start: pd.Timestamp, window_end: pd.Timestamp, count: int = 20000) -> pd.DataFrame:
    params = {
        "count": count,
        "find[dateString][$gte]": window_start.isoformat(),
        "find[dateString][$lte]": window_end.isoformat(),
    }
    entries = nightscout_get(
        cfg.nightscout_base_url,
        "/api/v1/entries/sgv.json",
        params=params,
        token=cfg.nightscout_token,
        api_secret=cfg.nightscout_api_secret,
    )
    if not entries:
        raise ValueError("No Nightscout glucose entries returned for this window.")

    glucose_df = pd.DataFrame(entries)
    if "date" in glucose_df.columns:
        glucose_df["timestamp"] = pd.to_datetime(glucose_df["date"], unit="ms", utc=True).dt.tz_convert(None)
    elif "dateString" in glucose_df.columns:
        glucose_df["timestamp"] = pd.to_datetime(glucose_df["dateString"], errors="coerce")
    else:
        raise ValueError("Nightscout entries missing date/dateString.")

    if "sgv" not in glucose_df.columns:
        raise ValueError("Nightscout entries missing sgv values.")

    return glucose_df[["timestamp", "sgv"]].dropna().sort_values("timestamp").reset_index(drop=True)


def _stream_values(stream_obj):
    if stream_obj is None:
        return None
    if isinstance(stream_obj, list):
        return stream_obj
    if isinstance(stream_obj, dict):
        if isinstance(stream_obj.get("data"), list):
            return stream_obj["data"]
        if isinstance(stream_obj.get("values"), list):
            return stream_obj["values"]
    return None


def normalize_streams(raw) -> Dict[str, List]:
    out: Dict[str, List] = {}

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = item.get("type") or item.get("name")
                values = _stream_values(item)
                if key and isinstance(values, list):
                    out[str(key)] = values
        return out

    if isinstance(raw, dict):
        if isinstance(raw.get("streams"), list):
            out.update(normalize_streams(raw["streams"]))

        for key, value in raw.items():
            values = _stream_values(value)
            if isinstance(values, list):
                out[str(key)] = values

    return out


def fetch_activity_streams(cfg: AnalysisConfig, activity_id: str, requested_streams: str = "time,heartrate,power") -> Tuple[dict, dict]:
    candidates: List[Tuple[str, str]] = []

    if cfg.intervals_streams_base_url:
        try:
            candidates.append(("url", cfg.intervals_streams_base_url.format(activity_id=activity_id)))
        except Exception:
            candidates.append(("url", cfg.intervals_streams_base_url))

    candidates.extend(
        [
            ("url", f"{cfg.intervals_base_url.rstrip('/')}/api/v1/activity/{activity_id}/streams.json"),
            ("url", f"{cfg.intervals_base_url.rstrip('/')}/api/v1/activities/{activity_id}/streams"),
            ("path", f"/api/v1/activity/{activity_id}/streams.json"),
            ("path", f"/api/v1/activities/{activity_id}/streams"),
        ]
    )

    param_variants = [
        {"streams": requested_streams},
        {"types": requested_streams},
        {"streams": requested_streams, "types": requested_streams},
        {},
    ]

    auth_variants = [
        ("bearer", {"Authorization": f"Bearer {cfg.intervals_api_key}"}, None),
        ("basic", {}, ("API_KEY", cfg.intervals_api_key)),
    ]

    attempts: List[str] = []

    for target_type, target in candidates:
        for params in param_variants:
            for auth_name, headers, basic_auth in auth_variants:
                try:
                    if target_type == "url":
                        resp = requests.get(target, headers=headers, params=params, auth=basic_auth, timeout=20)
                        resp.raise_for_status()
                        return resp.json(), {"target": target, "params": params, "auth": auth_name}

                    if auth_name != "basic":
                        continue

                    data = intervals_get_basic(cfg.intervals_base_url, cfg.intervals_api_key, target, params=params)
                    return data, {"target": target, "params": params, "auth": auth_name}
                except Exception as exc:
                    attempts.append(f"{target} | {params} | {auth_name}: {type(exc).__name__}")

    raise ValueError("Unable to fetch streams. Last attempts:\n" + "\n".join(attempts[-8:]))


def build_activity_timeseries(activity_start: pd.Timestamp, streams_map: Dict[str, List]) -> pd.DataFrame:
    hr_keys = ["heartrate", "hr", "heart_rate", "heartRate"]
    time_keys = ["time", "seconds", "sec"]
    power_keys = ["power", "watts", "watt"]

    hr_vals = next((streams_map[k] for k in hr_keys if k in streams_map and streams_map[k]), None)
    if not hr_vals:
        raise ValueError("No heart rate stream present in selected activity.")

    time_vals = next((streams_map[k] for k in time_keys if k in streams_map and streams_map[k]), None)
    if not time_vals:
        time_vals = list(range(len(hr_vals)))

    power_vals = next((streams_map[k] for k in power_keys if k in streams_map and streams_map[k]), None)

    n = min(len(hr_vals), len(time_vals))
    if power_vals:
        n = min(n, len(power_vals))

    df = pd.DataFrame(
        {
            "timestamp": activity_start + pd.to_timedelta(time_vals[:n], unit="s"),
            "hr": pd.to_numeric(pd.Series(hr_vals[:n]), errors="coerce"),
        }
    )

    if power_vals:
        df["power"] = pd.to_numeric(pd.Series(power_vals[:n]), errors="coerce")
    else:
        df["power"] = np.nan

    return df.dropna(subset=["hr"]).reset_index(drop=True)


def align_with_glucose(activity_df: pd.DataFrame, glucose_df: pd.DataFrame, tolerance: str = "3min") -> pd.DataFrame:
    bg = glucose_df.rename(columns={"sgv": "bg"}).sort_values("timestamp")[["timestamp", "bg"]]
    aligned = pd.merge_asof(
        activity_df.sort_values("timestamp"),
        bg,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    ).dropna(subset=["bg"])

    if aligned.empty:
        raise ValueError("No overlapping HR/BG samples after alignment.")
    return aligned


def bin_timeseries(aligned_df: pd.DataFrame, bin_size: str = "5min") -> pd.DataFrame:
    required_cols = ["timestamp", "hr", "bg"]
    missing = [c for c in required_cols if c not in aligned_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for binning: {missing}")

    work = aligned_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    agg_map = {"hr": "mean", "bg": "mean"}
    if "power" in work.columns:
        agg_map["power"] = "mean"

    binned = work.resample(bin_size).agg(agg_map).dropna(subset=["hr", "bg"]).reset_index()
    if "power" not in binned.columns:
        binned["power"] = np.nan

    if binned.empty:
        raise ValueError(f"No data left after {bin_size} binning.")
    return binned


def compute_correlation_metrics(aligned_df: pd.DataFrame, hi_quantile: float = 0.85) -> Tuple[pd.DataFrame, pd.Series]:
    out = aligned_df.copy()
    hr_threshold = out["hr"].quantile(hi_quantile)
    out["high_intensity"] = out["hr"] >= hr_threshold

    hi_df = out[out["high_intensity"]]

    pearson_hr_bg = out["hr"].corr(out["bg"], method="pearson")
    spearman_hr_bg = out["hr"].corr(out["bg"], method="spearman")
    pearson_hi_hr_bg = hi_df["hr"].corr(hi_df["bg"], method="pearson") if len(hi_df) > 2 else np.nan
    pearson_hi_flag_bg = out["high_intensity"].astype(int).corr(out["bg"], method="pearson")

    if "power" in out.columns and out["power"].notna().sum() > 2:
        pearson_power_bg = out["power"].corr(out["bg"], method="pearson")
    else:
        pearson_power_bg = np.nan

    per_min = out.set_index("timestamp")[["hr", "bg"]].resample("1min").mean().interpolate(limit_direction="both")
    lag_rows = []
    for lag in range(-60, 61, 5):
        shifted_bg = per_min["bg"].shift(-lag)
        lag_rows.append({"lag_minutes": lag, "pearson_hr_bg": per_min["hr"].corr(shifted_bg)})
    lag_df = pd.DataFrame(lag_rows)

    best_idx = lag_df["pearson_hr_bg"].abs().idxmax()
    best_lag = lag_df.iloc[best_idx]

    summary = pd.Series(
        {
            "samples_aligned": int(len(out)),
            "high_intensity_hr_threshold": float(hr_threshold),
            "pearson_hr_bg": float(pearson_hr_bg) if pd.notna(pearson_hr_bg) else np.nan,
            "spearman_hr_bg": float(spearman_hr_bg) if pd.notna(spearman_hr_bg) else np.nan,
            "pearson_hr_bg_high_intensity_only": float(pearson_hi_hr_bg) if pd.notna(pearson_hi_hr_bg) else np.nan,
            "pearson_high_intensity_flag_vs_bg": float(pearson_hi_flag_bg) if pd.notna(pearson_hi_flag_bg) else np.nan,
            "pearson_power_bg": float(pearson_power_bg) if pd.notna(pearson_power_bg) else np.nan,
            "best_abs_lag_minutes": int(best_lag["lag_minutes"]),
            "best_abs_lag_corr": float(best_lag["pearson_hr_bg"]) if pd.notna(best_lag["pearson_hr_bg"]) else np.nan,
        }
    )

    return lag_df, summary


def run_activity_analysis(
    cfg: AnalysisConfig,
    activity: dict,
    bin_size: str = "5min",
    hi_quantile: float = 0.85,
    requested_streams: str = "time,heartrate,power",
) -> Dict[str, object]:
    activity_id = activity["id"]
    activity_start, activity_end, window_start, window_end = get_activity_window(
        activity,
        pre_hours=cfg.pre_activity_hours,
        post_hours=cfg.post_activity_hours,
    )

    raw_streams, stream_meta = fetch_activity_streams(cfg, activity_id, requested_streams=requested_streams)
    streams_map = normalize_streams(raw_streams)
    activity_ts_df = build_activity_timeseries(activity_start, streams_map)
    activity_ts_df = activity_ts_df[
        (activity_ts_df["timestamp"] >= activity_start) & (activity_ts_df["timestamp"] <= activity_end)
    ]

    glucose_df = fetch_glucose(cfg, window_start, window_end)
    aligned_raw_df = align_with_glucose(activity_ts_df, glucose_df, tolerance=cfg.merge_tolerance)
    aligned_df = bin_timeseries(aligned_raw_df, bin_size=bin_size)
    lag_corr_df, corr_summary = compute_correlation_metrics(aligned_df, hi_quantile=hi_quantile)

    return {
        "activity_id": activity_id,
        "activity_start": activity_start,
        "activity_end": activity_end,
        "window_start": window_start,
        "window_end": window_end,
        "stream_meta": stream_meta,
        "stream_keys": sorted(streams_map.keys()),
        "activity_ts_df": activity_ts_df,
        "glucose_df": glucose_df,
        "aligned_raw_df": aligned_raw_df,
        "aligned_df": aligned_df,
        "lag_corr_df": lag_corr_df,
        "corr_summary": corr_summary,
    }


def plot_hr_bg_timeseries(aligned_df: pd.DataFrame, hi_quantile: float = 0.85, bin_size: str = "5min"):
    plot_df = aligned_df.copy()
    hr_hi_threshold = plot_df["hr"].quantile(hi_quantile)
    plot_df["high_intensity"] = plot_df["hr"] >= hr_hi_threshold

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(plot_df["timestamp"], plot_df["hr"], color="tab:red", linewidth=1.2, label="HR")
    ax1.scatter(
        plot_df.loc[plot_df["high_intensity"], "timestamp"],
        plot_df.loc[plot_df["high_intensity"], "hr"],
        color="darkred",
        s=8,
        alpha=0.7,
        label="High-intensity HR",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Heart Rate (bpm)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(plot_df["timestamp"], plot_df["bg"], color="tab:blue", linewidth=1.2, label="BG")
    ax2.set_ylabel("Glucose (mg/dL)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    ax1.set_title(f"HR and BG During Selected Activity ({bin_size} bins)")
    plt.tight_layout()
    plt.show()


def plot_scatter_correlations(aligned_df: pd.DataFrame, bin_size: str = "5min"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(aligned_df["hr"], aligned_df["bg"], alpha=0.5, s=10, label="HR vs BG")
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"Correlation Scatter: HR vs BG ({bin_size} bins)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    if "power" in aligned_df.columns and aligned_df["power"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(aligned_df["power"], aligned_df["bg"], alpha=0.5, s=10, label="Power vs BG", color="tab:purple")
        ax.set_xlabel("Power (W)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"Correlation Scatter: Power vs BG ({bin_size} bins)")
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_lag_correlation(lag_corr_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lag_corr_df["lag_minutes"], lag_corr_df["pearson_hr_bg"], marker="o", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    best_idx = lag_corr_df["pearson_hr_bg"].abs().idxmax()
    best_lag = int(lag_corr_df.loc[best_idx, "lag_minutes"])
    ax.axvline(best_lag, color="tab:green", linestyle="--", linewidth=1, label=f"Best lag: {best_lag} min")
    ax.set_title("Lagged Correlation: HR vs BG")
    ax.set_xlabel("Lag (minutes, + means BG delayed)")
    ax.set_ylabel("Pearson correlation")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_delay_adjusted_overlay(aligned_df: pd.DataFrame, lag_corr_df: pd.DataFrame, bin_size: str = "5min"):
    overlay_df = aligned_df.copy().sort_values("timestamp").reset_index(drop=True)
    best_idx = lag_corr_df["pearson_hr_bg"].abs().idxmax()
    best_lag = int(lag_corr_df.loc[best_idx, "lag_minutes"])
    lag_delta = pd.to_timedelta(best_lag, unit="min")

    bg_shifted = overlay_df[["timestamp", "bg"]].copy()
    bg_shifted["timestamp_shifted"] = bg_shifted["timestamp"] - lag_delta
    bg_shifted = bg_shifted.rename(columns={"bg": "bg_delay_adjusted"})

    overlay_df = pd.merge_asof(
        overlay_df[["timestamp", "hr", "power", "bg"]].sort_values("timestamp"),
        bg_shifted[["timestamp_shifted", "bg_delay_adjusted"]].sort_values("timestamp_shifted"),
        left_on="timestamp",
        right_on="timestamp_shifted",
        direction="nearest",
        tolerance=pd.Timedelta(bin_size),
    ).drop(columns=["timestamp_shifted"])

    def _zscore(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() < 2 or float(s.std()) == 0.0:
            return pd.Series([np.nan] * len(s), index=s.index)
        return (s - s.mean()) / s.std()

    overlay_df["hr_z"] = _zscore(overlay_df["hr"])
    overlay_df["bg_z"] = _zscore(overlay_df["bg"])
    overlay_df["bg_delay_z"] = _zscore(overlay_df["bg_delay_adjusted"])
    overlay_df["power_z"] = _zscore(overlay_df["power"]) if "power" in overlay_df.columns else np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(overlay_df["timestamp"], overlay_df["hr_z"], label="HR (z)", color="tab:red", linewidth=1.2)
    if overlay_df["power"].notna().any():
        ax.plot(overlay_df["timestamp"], overlay_df["power_z"], label="Power (z)", color="tab:green", linewidth=1.2)
    ax.plot(overlay_df["timestamp"], overlay_df["bg_z"], label="BG raw (z)", color="tab:blue", linewidth=1.0, alpha=0.6)
    ax.plot(overlay_df["timestamp"], overlay_df["bg_delay_z"], label="BG delay-adjusted (z)", color="navy", linewidth=1.8)
    ax.set_title(f"Delay-Adjusted Overlay ({bin_size} bins, best lag={best_lag} min)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized value (z-score)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return overlay_df


def build_shifted_bg_power_df(aligned_df: pd.DataFrame, lag_corr_df: pd.DataFrame, bin_size: str = "5min"):
    work = aligned_df.copy().sort_values("timestamp").reset_index(drop=True)
    best_idx = lag_corr_df["pearson_hr_bg"].abs().idxmax()
    best_lag = int(lag_corr_df.loc[best_idx, "lag_minutes"])
    lag_delta = pd.to_timedelta(best_lag, unit="min")

    # Shift BG by the estimated lag so BG is aligned with likely physiological response timing.
    bg_shifted = work[["timestamp", "bg"]].copy()
    bg_shifted["timestamp_shifted"] = bg_shifted["timestamp"] - lag_delta
    bg_shifted = bg_shifted.rename(columns={"bg": "bg_shifted"})

    # Nearest-time join keeps the binned cadence while attaching shifted BG values.
    gam_df = pd.merge_asof(
        work[["timestamp", "hr", "power"]].sort_values("timestamp"),
        bg_shifted[["timestamp_shifted", "bg_shifted"]].sort_values("timestamp_shifted"),
        left_on="timestamp",
        right_on="timestamp_shifted",
        direction="nearest",
        tolerance=pd.Timedelta(bin_size),
    ).drop(columns=["timestamp_shifted"])

    gam_df = gam_df.dropna(subset=["hr", "power", "bg_shifted"]).reset_index(drop=True)
    return gam_df, best_lag


def plot_gam_surface_and_fit(gam_df: pd.DataFrame, y_hat: np.ndarray, title_prefix: str = "GAM"):
    y = gam_df["power"].to_numpy()
    x = gam_df["hr"].to_numpy()
    b = gam_df["bg_shifted"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cf = axes[0].tricontourf(x, b, y_hat, levels=20, cmap="viridis")
    axes[0].set_xlabel("HR (bpm)")
    axes[0].set_ylabel("Shifted BG (mg/dL)")
    axes[0].set_title(f"{title_prefix} Predicted Power")
    cbar = plt.colorbar(cf, ax=axes[0])
    cbar.set_label("Predicted Power (W)")

    axes[1].scatter(y, y_hat, s=12, alpha=0.6)
    lim_min = float(min(np.nanmin(y), np.nanmin(y_hat)))
    lim_max = float(max(np.nanmax(y), np.nanmax(y_hat)))
    axes[1].plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
    axes[1].set_xlabel("Observed Power (W)")
    axes[1].set_ylabel("Predicted Power (W)")
    axes[1].set_title(f"{title_prefix} Observed vs Predicted")

    plt.tight_layout()
    plt.show()


def fit_power_gam(gam_df: pd.DataFrame):
    try:
        from pygam import LinearGAM, s, te
    except Exception as exc:
        raise ImportError("pygam not installed. Run: pip install pygam") from exc

    # Smooth + interaction terms model nonlinear HR/BG effects on power.
    X = gam_df[["hr", "bg_shifted"]].to_numpy()
    y = gam_df["power"].to_numpy()
    gam = LinearGAM(s(0) + s(1) + te(0, 1)).fit(X, y)
    y_hat = gam.predict(X)
    r2 = 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return gam, y_hat, float(r2)


def run_single_activity_gam(result: Dict[str, object], bin_size: str = "5min", min_samples: int = 20):
    aligned_df = result["aligned_df"]
    lag_corr_df = result["lag_corr_df"]

    if aligned_df["power"].notna().sum() < min_samples:
        return None

    gam_df, best_lag = build_shifted_bg_power_df(aligned_df, lag_corr_df, bin_size=bin_size)
    if len(gam_df) < min_samples:
        return None

    _, y_hat, r2 = fit_power_gam(gam_df)
    return {
        "gam_df": gam_df,
        "y_hat": y_hat,
        "r2": r2,
        "best_lag": best_lag,
        "samples": int(len(gam_df)),
    }


def run_pooled_gam(
    cfg: AnalysisConfig,
    activities_df: pd.DataFrame,
    bin_size: str = "5min",
    hi_quantile: float = 0.85,
    max_activities: int = 30,
    min_samples_per_activity: int = 20,
):
    recent_df = activities_df.head(max_activities).reset_index(drop=True)
    pooled_parts = []
    used_activities = []

    # Build pooled training data from activities that have enough usable power + shifted BG samples.
    for i in range(len(recent_df)):
        activity_row = recent_df.iloc[i].to_dict()
        try:
            res_i = run_activity_analysis(
                cfg=cfg,
                activity=activity_row,
                bin_size=bin_size,
                hi_quantile=hi_quantile,
                requested_streams="time,heartrate,power",
            )
        except Exception:
            continue

        aligned_i = res_i["aligned_df"]
        if aligned_i["power"].notna().sum() < min_samples_per_activity:
            continue

        gam_i, best_lag_i = build_shifted_bg_power_df(aligned_i, res_i["lag_corr_df"], bin_size=bin_size)
        if len(gam_i) < min_samples_per_activity:
            continue

        gam_i["activity_id"] = str(res_i["activity_id"])
        gam_i["best_lag_minutes"] = best_lag_i
        pooled_parts.append(gam_i)
        used_activities.append(str(res_i["activity_id"]))

    if not pooled_parts:
        return None

    pooled_gam_df = pd.concat(pooled_parts, ignore_index=True)
    _, y_hat, r2 = fit_power_gam(pooled_gam_df)
    return {
        "pooled_gam_df": pooled_gam_df,
        "y_hat": y_hat,
        "r2": r2,
        "used_activities": sorted(set(used_activities)),
        "samples": int(len(pooled_gam_df)),
    }


def compute_best_bg_by_hr_bin(
    eff_pool_df: pd.DataFrame,
    hr_bin_size: int = 10,
):
    if eff_pool_df.empty:
        return pd.DataFrame()

    work = eff_pool_df.copy()
    work = work.dropna(subset=["hr", "power", "bg_shifted"])
    work = work[work["hr"] > 0]
    if work.empty:
        return pd.DataFrame()

    # Efficiency proxy: watts produced per heartbeat.
    work["efficiency"] = work["power"] / work["hr"]
    hr_min = int(np.floor(work["hr"].min() / float(hr_bin_size)) * hr_bin_size)
    hr_max = int(np.ceil(work["hr"].max() / float(hr_bin_size)) * hr_bin_size)
    bins = np.arange(hr_min, hr_max + hr_bin_size, hr_bin_size)
    work["hr_bin"] = pd.cut(work["hr"], bins=bins, right=False)

    grp = work.groupby("hr_bin", observed=True)
    idx_best = grp["efficiency"].idxmax()
    best = work.loc[idx_best, ["hr_bin", "hr", "bg_shifted", "efficiency", "power"]].copy()
    best = best.rename(columns={"bg_shifted": "best_bg_shifted"}).sort_values("hr").reset_index(drop=True)
    return best


def plot_best_bg_by_hr_bin(best_bg_by_bin: pd.DataFrame, title_suffix: str = ""):
    if best_bg_by_bin.empty:
        print("No data to plot for best BG by HR bin.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(best_bg_by_bin["hr"], best_bg_by_bin["best_bg_shifted"], marker="o", color="tab:blue", label="Best BG (shifted)")
    ax1.set_xlabel("HR (bpm, per 10-bpm bin representative)")
    ax1.set_ylabel("Best shifted BG (mg/dL)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(best_bg_by_bin["hr"], best_bg_by_bin["efficiency"], marker="s", color="tab:green", label="Power/HR efficiency")
    ax2.set_ylabel("Efficiency (W per bpm)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    title = "Most Efficient Shifted BG by 10-bpm HR Bin"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax1.set_title(title)
    plt.tight_layout()
    plt.show()
