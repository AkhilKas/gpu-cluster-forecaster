"""
Handle user-uploaded CSVs of GPU telemetry.

Flow per machine:
    1. Validate columns (cpu_usage + memory_usage required; assigned_memory
       and cycles_per_instruction optional, filled with 0 if missing).
    2. Fit a fresh MinMaxScaler on the uploaded rows for that machine.
    3. Take the last `sequence_length` rows as the input window.
    4. Run the model.
    5. Denormalize the forecast back to real units using the fresh scaler.

The model was trained on per-machine MinMaxScalers, so refitting on the
user's data mimics that behavior. If the user's data distribution is wildly
different from the training range, the forecast will still be usable but
less accurate — we don't detect that automatically.
"""

import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.preprocessing import MinMaxScaler

from app.api.schemas import HistoryPoint, UploadedMachine, UploadPredictResponse
from app.config import DataConfig

from .model_registry import ModelRegistry
from .predictor import Predictor

logger = logging.getLogger(__name__)


class UploadPredictService:
    """Turn an uploaded DataFrame into per-machine forecasts."""

    REQUIRED_COLS = ("cpu_usage", "memory_usage")

    def __init__(
        self,
        predictor: Predictor,
        registry: ModelRegistry,
        config: DataConfig,
    ):
        self.predictor = predictor
        self.registry = registry
        self.config = config

    def process(
        self, df: pd.DataFrame, model_name: str | None = None
    ) -> UploadPredictResponse:
        # Resolve the model up front so we fail fast if none is loaded.
        resolved = model_name or self.registry.default_name()
        if resolved is None:
            raise HTTPException(status_code=503, detail="No models loaded.")
        if resolved not in self.registry.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {resolved!r} not found or not invokable.",
            )
        model = self.registry.get(resolved)
        expected_input_dim = getattr(model, "input_dim", None)

        seq_len = self.config.sequence_length
        target_cols = list(self.config.target_columns)

        # Validate required columns.
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required column(s): {missing}. "
                f"Need at least {list(self.REQUIRED_COLS)}.",
            )

        top_warnings: list[str] = []

        # Coerce all known feature columns to numeric (skip missing).
        for col in self.config.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Match the training-time feature ordering + count to the model's
        # actual input_dim. Trims extras, pads missing with 0.
        feature_cols = self._resolve_feature_cols(df, expected_input_dim, top_warnings)

        # Group by machine_id, or treat everything as a single machine.
        if "machine_id" in df.columns:
            groups = [(str(mid), g) for mid, g in df.groupby("machine_id", sort=True)]
        else:
            groups = [("uploaded", df)]
            top_warnings.append(
                "No 'machine_id' column; treating all rows as one machine."
            )

        machines_out: list[UploadedMachine] = []
        for mid, group in groups:
            machines_out.append(
                self._process_one_machine(
                    mid, group, resolved, seq_len, feature_cols, target_cols
                )
            )

        return UploadPredictResponse(
            model=resolved,
            num_machines=len(machines_out),
            target_columns=target_cols,
            machines=machines_out,
            warnings=top_warnings,
        )

    def _resolve_feature_cols(
        self,
        df: pd.DataFrame,
        expected_dim: int | None,
        top_warnings: list[str],
    ) -> list[str]:
        """
        Pick the feature-column subset the model actually expects.

        Mirrors `Preprocessor.process_machine`: take config.feature_columns in
        their configured order, keep the ones present in the DataFrame, then
        adjust to the model's `input_dim` by trimming extras or padding zeros.
        """
        df_cols = set(df.columns)
        present = [c for c in self.config.feature_columns if c in df_cols]

        if expected_dim is None:
            return present

        if len(present) > expected_dim:
            dropped = present[expected_dim:]
            present = present[:expected_dim]
            top_warnings.append(
                f"Model expects {expected_dim} features; ignoring extras: {dropped}."
            )
            return present

        if len(present) < expected_dim:
            missing_from_config = [
                c for c in self.config.feature_columns if c not in df_cols
            ]
            need = expected_dim - len(present)
            for col in missing_from_config[:need]:
                df[col] = 0.0
                top_warnings.append(f"Column '{col}' missing; filled with 0.")
            # Re-derive `present` so it's in config order.
            df_cols = set(df.columns)
            present = [c for c in self.config.feature_columns if c in df_cols][
                :expected_dim
            ]

        return present

    def _process_one_machine(
        self,
        mid: str,
        group: pd.DataFrame,
        model_name: str,
        seq_len: int,
        feature_cols: list[str],
        target_cols: list[str],
    ) -> UploadedMachine:
        warnings: list[str] = []

        # Drop rows with missing required values.
        group = group.dropna(subset=list(self.REQUIRED_COLS))
        n_rows = len(group)

        if n_rows < seq_len:
            warnings.append(
                f"Not enough rows: {n_rows} available, {seq_len} required for a forecast."
            )
            return UploadedMachine(
                machine_id=str(mid),
                num_input_rows=n_rows,
                history=[],
                forecast=[],
                warnings=warnings,
            )

        # Fit a fresh scaler on this machine's data.
        raw = group[feature_cols].values.astype(np.float32)
        scaler = MinMaxScaler()
        scaler.fit(raw)

        # Guard against fully-constant columns (scale_ == 0 blows up denorm).
        if np.any(scaler.scale_ == 0):
            warnings.append(
                "One or more feature columns had constant values; forecast "
                "may be degenerate."
            )

        normalized = scaler.transform(raw)
        window = normalized[-seq_len:]

        # Model call: (seq_len, n_features) → (horizon, n_targets), normalized.
        forecast_norm = self.predictor.predict(window, model_name)

        # Denormalize history (already in real units — just report last seq_len rows).
        history_real = raw[-seq_len:]

        # Denormalize forecast targets. If the model's targets aren't all in
        # our feature_cols (unusual), fall back to reporting the normalized
        # values so the caller still gets something.
        target_in_features = [c for c in target_cols if c in feature_cols]
        if not target_in_features:
            forecast_real = forecast_norm
            warnings.append(
                "Target columns not present in resolved features; forecast is "
                "returned in normalized [0,1] space."
            )
        else:
            n_features = len(feature_cols)
            dummy = np.zeros((forecast_norm.shape[0], n_features), dtype=np.float32)
            for i, col in enumerate(target_cols):
                if col in feature_cols:
                    dummy[:, feature_cols.index(col)] = forecast_norm[:, i]
            inversed = scaler.inverse_transform(dummy)
            target_idx = [feature_cols.index(c) for c in target_in_features]
            forecast_real = inversed[:, target_idx]

        history = [
            HistoryPoint(
                step=i,
                values={
                    col: float(history_real[i, j]) for j, col in enumerate(feature_cols)
                },
            )
            for i in range(history_real.shape[0])
        ]
        forecast = [
            HistoryPoint(
                step=i,
                values={
                    col: float(forecast_real[i, j]) for j, col in enumerate(target_cols)
                },
            )
            for i in range(forecast_real.shape[0])
        ]

        return UploadedMachine(
            machine_id=str(mid),
            num_input_rows=n_rows,
            history=history,
            forecast=forecast,
            warnings=warnings,
        )
