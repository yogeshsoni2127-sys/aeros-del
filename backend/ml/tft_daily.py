"""
TFT Daily Member — live inference for research checkpoints.

Runs the SIH-p2 TemporalFusionTransformer checkpoints (1-step daily,
14-day encoder, GroupNormalizer per location) inside the live service.
Mirrors scripts/training/train_tft.py dataset construction exactly:

- group location_name / dense-rank time_idx / max_encoder_length=14 /
  max_prediction_length=1
- known reals: day_of_week, month, is_stubble_season, wind_toward_delhi,
  is_holiday, is_diwali_week, o3_driver
- unknown reals: 24-col base + blh_max/radiation_sum/dewpoint_mean/
  ventilation + the target itself
- GroupNormalizer(groups=["location_name"]), add_relative_time_idx /
  add_target_scales / add_encoder_length, allow_missing_timesteps

Two documented deviations from training (validated by the gate in
scripts/validate_tft_gate.py — run it after any change here):
1. Normalizer refit on the inference window (fitted scales were never
   saved; only hparams.yaml survived). Validated harmless on seed replay.
2. Single shared batch over ALL locations per (target, horizon) so the
   location_name categorical encoding matches training (per-station
   datasets would reindex the embedding and silently scramble stations).

Recursive horizons (h=2,3): encoder slides forward on prior preds, same
as the tree members. Missing TFT (no torch/forecasting, no ckpt) keeps
the research offline fallback (tft := xgb ?? lgbm) via the caller.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TARGETS = ["pm25", "pm10", "no2", "o3"]
ENCODER_LENGTH = 14

KNOWN_REALS = [
    "day_of_week", "month", "is_stubble_season", "wind_toward_delhi",
    "is_holiday", "is_diwali_week", "o3_driver",
]

UNKNOWN_REALS_BASE = [
    "regional_fire_count", "regional_total_frp", "regional_max_frp",
    "regional_fire_count_lag1", "regional_fire_count_lag2",
    "regional_total_frp_lag1", "regional_total_frp_lag3",
    "regional_fire_count_roll3", "regional_fire_count_roll7",
    "regional_total_frp_roll3", "regional_total_frp_roll7",
    "local_fire_count", "local_total_frp", "local_max_frp",
    "temp_mean", "temp_max", "temp_min", "humidity_mean",
    "wind_speed_mean", "wind_direction_dominant", "precipitation",
    "pressure_mean", "wind_gusts_mean", "fire_wind_interaction",
]

UNKNOWN_REALS_FIX1 = ["blh_max", "radiation_sum", "dewpoint_mean",
                      "ventilation"]

# Best-validation-loss checkpoint version per target (from
# models/tft_checkpoints/*/lightning_logs/*/metrics.csv minima).
BEST_VERSION = {
    "pm25": "version_1",
    "pm10": "version_0",
    "no2": "version_0",
    "o3": "version_1",
}


class TFTDailyEnsemble:
    """Per-target TFT checkpoints, batched multi-station inference."""

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.models: Dict[str, Any] = {}
        # Per-target covariate lists read from each checkpoint's own
        # dataset_parameters (the four models were trained on an older
        # CSV schema: 4 known + 24 unknown + target, no Fix-1/optional
        # columns). Never hardcode: the checkpoint describes itself.
        self.known_reals: Dict[str, List[str]] = {}
        self.unknown_reals: Dict[str, List[str]] = {}

    def load(self) -> bool:
        """Load one vendored checkpoint per target. True if >= 1 loads."""
        try:
            import torch  # noqa: F401
            from pytorch_forecasting import TemporalFusionTransformer
        except ImportError as e:
            logger.warning("tft-daily: torch/forecasting missing (%s)", e)
            return False
        for t in TARGETS:
            ckpt = os.path.join(self.ckpt_dir, f"tft_{t}.ckpt")
            if not os.path.exists(ckpt):
                logger.warning("tft-daily: no checkpoint for %s", t)
                continue
            try:
                import torch
                model = TemporalFusionTransformer.load_from_checkpoint(
                    ckpt, map_location="cpu")
                model.eval()
                self.models[t] = model
                dp = (model.hparams.get("dataset_parameters", {})
                      if hasattr(model, "hparams") else {}) or {}
                self.known_reals[t] = list(
                    dp.get("time_varying_known_reals") or KNOWN_REALS)
                self.unknown_reals[t] = list(
                    dp.get("time_varying_unknown_reals")
                    or (UNKNOWN_REALS_BASE + UNKNOWN_REALS_FIX1 + [t]))
            except Exception as e:
                logger.warning("tft-daily: load %s failed: %s", t, e)
        if self.models:
            logger.info("tft-daily loaded: %s",
                        ",".join(sorted(self.models)))
        return bool(self.models)

    @property
    def available(self) -> bool:
        return bool(self.models)

    def runnable_targets(self) -> List[str]:
        return [t for t in TARGETS if t in self.models]

    # ── Inference ────────────────────────────────────────────────

    def predict_horizon(
        self,
        target: str,
        frames: Dict[str, List[dict]],
    ) -> Dict[str, Optional[float]]:
        """One batched forward pass for a single target+horizon.

        frames: {location: [14 encoder dicts + 1 decoder dict]}.
        Encoder dicts carry `<target>_actual` columns (one per
        pollutant); the active target column is materialised here.
        Covariate NaNs are filled with per-station trailing medians
        (TFT has no native missing-covariate handling; trees keep the
        NaN path in the caller's feat dicts, untouched by this).
        Returns {location: Day prediction} (None per station on failure
        — caller keeps the tree-only fallback there).
        """
        out: Dict[str, Optional[float]] = {loc: None for loc in frames}
        model = self.models.get(target)
        if model is None or not frames:
            return out
        try:
            import pandas as pd
            import torch
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            return out
        try:
            filled = {}
            for loc, fr in frames.items():
                if len(fr) != ENCODER_LENGTH + 1:
                    continue
                med = {}
                for c in set().union(*(set(r.keys()) for r in fr)):
                    if c in ("location_name", "date", "time_idx"):
                        continue
                    vals = []
                    for r in fr:
                        try:
                            f = float(r.get(c))
                            if math.isfinite(f):
                                vals.append(f)
                        except (TypeError, ValueError):
                            pass
                    if vals:
                        vals.sort()
                        med[c] = vals[len(vals) // 2]
                rows = []
                for r in fr:
                    row = dict(r)
                    for c, m in med.items():
                        try:
                            f = float(row.get(c))
                            if not math.isfinite(f):
                                row[c] = m
                        except (TypeError, ValueError):
                            row[c] = m
                    rows.append(row)
                # Decoder target is never a model input at predict time,
                # but the dataset rejects NaN targets outright: persist
                # the last encoder actual (harmless placeholder).
                try:
                    last = None
                    for r in rows[-2::-1]:
                        try:
                            f = float(r.get(target))
                            if math.isfinite(f):
                                last = f
                                break
                        except (TypeError, ValueError):
                            continue
                    if last is not None:
                        try:
                            f = float(rows[-1].get(target))
                            if not math.isfinite(f):
                                rows[-1][target] = last
                        except (TypeError, ValueError):
                            rows[-1][target] = last
                except Exception:
                    pass
                filled[loc] = rows
            if not filled:
                return out
            rows = []
            for loc, fr in filled.items():
                for i, r in enumerate(fr):
                    row = dict(r)
                    row["location_name"] = loc
                    row["time_idx"] = i
                    try:
                        av = float(r.get(f"{target}_actual"))
                        row[target] = av if math.isfinite(av) else float("nan")
                    except (TypeError, ValueError):
                        row[target] = float("nan")
                    rows.append(row)
            if not rows:
                return out
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            known = [c for c in self.known_reals.get(target, KNOWN_REALS)
                     if c in df.columns]
            ulist = list(self.unknown_reals.get(
                target, UNKNOWN_REALS_BASE + UNKNOWN_REALS_FIX1 + [target]))
            if target not in ulist:
                ulist.append(target)
            unknown = [c for c in ulist if c in df.columns]
            params = dict(
                time_idx="time_idx",
                target=target,
                group_ids=["location_name"],
                max_encoder_length=ENCODER_LENGTH,
                max_prediction_length=1,
                static_categoricals=["location_name"],
                time_varying_known_reals=known,
                time_varying_unknown_reals=unknown,
                target_normalizer=GroupNormalizer(groups=["location_name"]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )
            # forecasting>=1.0 builds prediction windows via from_dataset
            # (the __init__ predict flag is gone); one window per group.
            base_ds = TimeSeriesDataSet(df, **params)
            ds = TimeSeriesDataSet.from_dataset(
                base_ds, df, predict=True, stop_randomization=True)
            dl = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
            with torch.no_grad():
                # logger/checkpointing off: predict() otherwise litters
                # lightning_logs/ version dirs on every refresh.
                res = model.predict(
                    dl, mode="prediction", return_index=True,
                    trainer_kwargs={"logger": False,
                                    "enable_checkpointing": False})
            raw = res["output"] if isinstance(res, dict) else res.output
            idx = None
            try:
                idx = res["index"] if isinstance(res, dict) else res.index
            except Exception:
                idx = None
            preds = raw.detach().cpu().numpy().reshape(-1)
            names = self._index_names(ds, idx, len(preds))
            for loc_name, pv in zip(names, preds):
                if loc_name in out and out[loc_name] is None:
                    try:
                        f = float(pv)
                        out[loc_name] = f if math.isfinite(f) else None
                    except (TypeError, ValueError):
                        pass
            return out
        except Exception as e:
            logger.debug("tft-daily predict %s failed: %s", target, e)
            return out

    @staticmethod
    def _index_names(ds, idx, n):
        """Decode prediction rows to location names.

        Prefers the returned index frame; falls back to the dataset's
        group order (predict=True + no shuffle emits one window per
        group in group order — the gate validates this mapping).
        """
        try:
            import pandas as pd
            if isinstance(idx, pd.DataFrame) and len(idx) == n:
                for col in ("location_name", "group_location_name",
                            "groups"):
                    if col in idx.columns:
                        return [str(v) for v in idx[col].tolist()]
                return [str(v) for v in idx.iloc[:, 0].tolist()]
        except Exception:
            pass
        try:
            groups = list(ds.get_parameters().get("group_ids", [])) \
                if hasattr(ds, "get_parameters") else []
            _ = groups
        except Exception:
            pass
        return [None] * n
