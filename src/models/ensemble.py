from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class BlendedIsotonicCalibrator:
    """Isotonic calibration blended back toward raw scores to preserve ranking resolution."""

    def __init__(self, blend: float = 0.7) -> None:
        self.blend = blend
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "BlendedIsotonicCalibrator":
        self._iso.fit(raw, y)
        self._fitted = True
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return raw
        calibrated = self._iso.transform(raw)
        result = self.blend * calibrated + (1.0 - self.blend) * raw
        return np.clip(result, 0.0, 1.0)


class StackedEnsemble:
    """
    Two-level stacked ensemble:
      Level 0: LightGBM + XGBoost (trained on OOF folds)
      Level 1: Logistic regression meta-learner
      Post-processing: BlendedIsotonicCalibrator
    """

    def __init__(
        self,
        lgbm_params: dict[str, Any] | None = None,
        xgb_params: dict[str, Any] | None = None,
        n_folds: int = 5,
        random_seed: int = 42,
        calibrator_blend: float = 0.7,
    ) -> None:
        self.lgbm_params = lgbm_params or {}
        self.xgb_params = xgb_params or {}
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.calibrator_blend = calibrator_blend

        self._lgbm_models: list[Any] = []
        self._xgb_models: list[Any] = []
        self._meta: LogisticRegression | None = None
        self._calibrator: BlendedIsotonicCalibrator | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "StackedEnsemble":
        self.feature_names_ = list(x_train.columns)
        x_arr = x_train.values.astype(float)
        y_arr = y_train.astype(int)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        oof_lgbm = np.zeros(len(x_arr))
        oof_xgb = np.zeros(len(x_arr))

        self._lgbm_models = []
        self._xgb_models = []

        for fold_train_idx, fold_val_idx in skf.split(x_arr, y_arr):
            xf_tr, xf_val = x_arr[fold_train_idx], x_arr[fold_val_idx]
            yf_tr, yf_val = y_arr[fold_train_idx], y_arr[fold_val_idx]

            lgbm_model = self._fit_lgbm(xf_tr, yf_tr, xf_val, yf_val, self.feature_names_)
            self._lgbm_models.append(lgbm_model)
            oof_lgbm[fold_val_idx] = self._predict_lgbm(lgbm_model, xf_val)

            xgb_model = self._fit_xgb(xf_tr, yf_tr, xf_val, yf_val)
            self._xgb_models.append(xgb_model)
            oof_xgb[fold_val_idx] = self._predict_xgb(xgb_model, xf_val)

        # Meta-learner on OOF predictions
        meta_x = np.column_stack([oof_lgbm, oof_xgb])
        self._meta = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_seed)
        self._meta.fit(meta_x, y_arr)

        # Calibration on OOF meta-predictions
        meta_scores = self._meta.predict_proba(meta_x)[:, 1]
        self._calibrator = BlendedIsotonicCalibrator(blend=self.calibrator_blend)
        self._calibrator.fit(meta_scores, y_arr)

        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        x_arr = features.reindex(columns=self.feature_names_, fill_value=0.0).values.astype(float)
        lgbm_scores = np.mean([self._predict_lgbm(m, x_arr) for m in self._lgbm_models], axis=0)
        xgb_scores = np.mean([self._predict_xgb(m, x_arr) for m in self._xgb_models], axis=0)
        meta_x = np.column_stack([lgbm_scores, xgb_scores])
        meta_scores = self._meta.predict_proba(meta_x)[:, 1]  # type: ignore[union-attr]
        return self._calibrator.transform(meta_scores)  # type: ignore[union-attr]

    def feature_importance(self, feature_names: list[str]) -> list[dict[str, Any]]:
        if not self._lgbm_models:
            return []
        importances = np.zeros(len(feature_names))
        for model in self._lgbm_models:
            imp = getattr(model, "feature_importances_", None)
            if imp is not None and len(imp) == len(feature_names):
                importances += imp
        importances /= max(len(self._lgbm_models), 1)
        pairs = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
        return [{"feature": name, "importance": float(score)} for name, score in pairs]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StackedEnsemble":
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_lgbm(self, x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, feature_names: list[str] | None = None) -> Any:
        import pandas as pd
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        params = dict(self.lgbm_params)
        early = int(params.pop("early_stopping_rounds", 50) or 50)
        params.setdefault("random_state", self.random_seed)
        model = LGBMClassifier(**params)
        callbacks = [early_stopping(early, verbose=False), log_evaluation(period=0)]
        # Pass DataFrames so LightGBM stores feature names internally and predict won't warn.
        df_tr = pd.DataFrame(x_tr, columns=feature_names) if feature_names else x_tr
        df_val = pd.DataFrame(x_val, columns=feature_names) if feature_names else x_val
        if len(set(y_val.tolist())) > 1:
            model.fit(df_tr, y_tr, eval_set=[(df_val, y_val)], callbacks=callbacks)
        else:
            model.fit(df_tr, y_tr)
        return model

    def _predict_lgbm(self, model: Any, x: np.ndarray) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame(x, columns=self.feature_names_) if self.feature_names_ else x
        proba = model.predict_proba(df)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _fit_xgb(self, x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Any:
        from xgboost import XGBClassifier

        params = dict(self.xgb_params)
        params.setdefault("random_state", self.random_seed)
        params.setdefault("eval_metric", "auc")
        params.setdefault("verbosity", 0)
        early = int(params.pop("early_stopping_rounds", 50) or 50)
        params["early_stopping_rounds"] = early
        model = XGBClassifier(**params)
        if len(set(y_val.tolist())) > 1:
            model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
        else:
            model.fit(x_tr, y_tr)
        return model

    def _predict_xgb(self, model: Any, x: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 else proba
