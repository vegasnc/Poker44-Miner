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


class FPRConstrainedCalibrator:
    """Grid-search bias+temperature logit transform to hit target FPR.

    Applies: score = sigmoid((logit(raw) + bias) / temperature)
    Grid-searches bias ∈ [-2, 2] and temperature ∈ [0.4, 2.5] to maximise
    average precision while keeping FPR ≤ target_fpr.
    Falls back to identity if no valid config found.
    """

    def __init__(self, target_fpr: float = 0.04, threshold: float = 0.5) -> None:
        self.target_fpr = target_fpr
        self.threshold = threshold
        self.bias_: float = 0.0
        self.temperature_: float = 1.0
        self._fitted = False

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "FPRConstrainedCalibrator":
        from sklearn.metrics import average_precision_score
        import logging

        eps = 1e-6
        raw_c = np.clip(raw, eps, 1 - eps)
        logit_raw = np.log(raw_c / (1 - raw_c))
        human_mask = y == 0

        best_ap, best_bias, best_temp = -1.0, 0.0, 1.0

        for bias in np.linspace(-2.0, 2.0, 17):
            for temp in np.linspace(0.4, 2.5, 15):
                scores = 1.0 / (1.0 + np.exp(-((logit_raw + bias) / temp)))
                preds = (scores >= self.threshold).astype(int)
                if human_mask.sum() > 0:
                    fpr = preds[human_mask].mean()
                    if fpr > self.target_fpr:
                        continue
                ap = average_precision_score(y, scores)
                if ap > best_ap:
                    best_ap, best_bias, best_temp = ap, bias, temp

        self.bias_ = best_bias
        self.temperature_ = best_temp
        self._fitted = True
        logging.getLogger(__name__).info(
            "FPR calibrator: bias=%.3f temp=%.3f AP=%.4f", best_bias, best_temp, best_ap
        )
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return raw
        eps = 1e-6
        raw_c = np.clip(raw, eps, 1 - eps)
        logit_raw = np.log(raw_c / (1 - raw_c))
        scores = 1.0 / (1.0 + np.exp(-((logit_raw + self.bias_) / self.temperature_)))
        return np.clip(scores, 0.0, 1.0)


class StackedEnsemble:
    """
    Two-level stacked ensemble:
      Level 0: LightGBM + XGBoost + CatBoost + Set Transformer (OOF folds)
      Level 1: Logistic regression meta-learner
      Post-processing: BlendedIsotonicCalibrator → FPRConstrainedCalibrator

    Sample weights: human chunks upweighted (human_weight, default 2.0) to
    protect against false positives, which the validator penalises quadratically.
    """

    def __init__(
        self,
        lgbm_params: dict[str, Any] | None = None,
        xgb_params: dict[str, Any] | None = None,
        catboost_params: dict[str, Any] | None = None,
        set_transformer_params: dict[str, Any] | None = None,
        n_folds: int = 5,
        random_seed: int = 42,
        calibrator_blend: float = 0.7,
        human_weight: float = 2.0,
        target_fpr: float = 0.04,
        use_set_transformer: bool = True,
    ) -> None:
        self.lgbm_params = lgbm_params or {}
        self.xgb_params = xgb_params or {}
        self.catboost_params = catboost_params or {}
        self.set_transformer_params = set_transformer_params or {}
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.calibrator_blend = calibrator_blend
        self.human_weight = human_weight
        self.target_fpr = target_fpr
        self.use_set_transformer = use_set_transformer

        self._lgbm_models: list[Any] = []
        self._xgb_models: list[Any] = []
        self._cat_models: list[Any] = []
        self._st_models: list[Any] = []
        self._meta: LogisticRegression | None = None
        self._iso_calibrator: BlendedIsotonicCalibrator | None = None
        self._fpr_calibrator: FPRConstrainedCalibrator | None = None
        self.feature_names_: list[str] = []
        self._has_st: bool = False

    def _make_weights(self, y: np.ndarray) -> np.ndarray:
        w = np.where(y == 0, self.human_weight, 1.0).astype(float)
        return w / w.mean()

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        hand_matrices_train: list[np.ndarray] | None = None,
        hand_matrices_valid: list[np.ndarray] | None = None,
    ) -> "StackedEnsemble":
        self.feature_names_ = list(x_train.columns)
        x_arr = x_train.values.astype(float)
        y_arr = y_train.astype(int)
        sample_weights = self._make_weights(y_arr)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        oof_lgbm = np.zeros(len(x_arr))
        oof_xgb = np.zeros(len(x_arr))
        oof_cat = np.zeros(len(x_arr))
        oof_st = np.zeros(len(x_arr))

        self._lgbm_models = []
        self._xgb_models = []
        self._cat_models = []
        self._st_models = []
        self._has_st = False

        has_matrices = (
            self.use_set_transformer
            and hand_matrices_train is not None
            and len(hand_matrices_train) == len(y_arr)
        )

        for fold_train_idx, fold_val_idx in skf.split(x_arr, y_arr):
            xf_tr, xf_val = x_arr[fold_train_idx], x_arr[fold_val_idx]
            yf_tr, yf_val = y_arr[fold_train_idx], y_arr[fold_val_idx]
            wf_tr = sample_weights[fold_train_idx]

            lgbm_model = self._fit_lgbm(xf_tr, yf_tr, xf_val, yf_val, wf_tr)
            self._lgbm_models.append(lgbm_model)
            oof_lgbm[fold_val_idx] = self._predict_lgbm(lgbm_model, xf_val)

            xgb_model = self._fit_xgb(xf_tr, yf_tr, xf_val, yf_val, wf_tr)
            self._xgb_models.append(xgb_model)
            oof_xgb[fold_val_idx] = self._predict_xgb(xgb_model, xf_val)

            cat_model = self._fit_catboost(xf_tr, yf_tr, xf_val, yf_val, wf_tr)
            self._cat_models.append(cat_model)
            oof_cat[fold_val_idx] = self._predict_catboost(cat_model, xf_val)

            if has_matrices:
                mf_tr = [hand_matrices_train[i] for i in fold_train_idx]
                mf_val = [hand_matrices_train[i] for i in fold_val_idx]
                st_model = self._fit_st(mf_tr, yf_tr, mf_val, yf_val, wf_tr)
                if st_model is not None:
                    self._st_models.append(st_model)
                    oof_st[fold_val_idx] = self._predict_st(st_model, mf_val)
                    self._has_st = True

        # Meta-learner
        if self._has_st:
            meta_x = np.column_stack([oof_lgbm, oof_xgb, oof_cat, oof_st])
        else:
            meta_x = np.column_stack([oof_lgbm, oof_xgb, oof_cat])

        self._meta = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_seed)
        self._meta.fit(meta_x, y_arr, sample_weight=sample_weights)

        # Stage 1: Blended isotonic calibration
        meta_scores = self._meta.predict_proba(meta_x)[:, 1]
        self._iso_calibrator = BlendedIsotonicCalibrator(blend=self.calibrator_blend)
        self._iso_calibrator.fit(meta_scores, y_arr)

        # Stage 2: FPR-constrained logit transform
        iso_scores = self._iso_calibrator.transform(meta_scores)
        self._fpr_calibrator = FPRConstrainedCalibrator(target_fpr=self.target_fpr)
        self._fpr_calibrator.fit(iso_scores, y_arr)

        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
        hand_matrices: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        x_arr = features.reindex(columns=self.feature_names_, fill_value=0.0).values.astype(float)
        lgbm_scores = np.mean([self._predict_lgbm(m, x_arr) for m in self._lgbm_models], axis=0)
        xgb_scores = np.mean([self._predict_xgb(m, x_arr) for m in self._xgb_models], axis=0)
        cat_scores = (
            np.mean([self._predict_catboost(m, x_arr) for m in self._cat_models], axis=0)
            if self._cat_models else xgb_scores
        )

        if self._has_st and self._st_models and hand_matrices is not None:
            st_scores = np.mean(
                [self._predict_st(m, hand_matrices) for m in self._st_models], axis=0
            )
            meta_x = np.column_stack([lgbm_scores, xgb_scores, cat_scores, st_scores])
        else:
            meta_x = np.column_stack([lgbm_scores, xgb_scores, cat_scores])

        meta_scores = self._meta.predict_proba(meta_x)[:, 1]  # type: ignore[union-attr]
        iso_scores = self._iso_calibrator.transform(meta_scores)      # type: ignore[union-attr]
        return self._fpr_calibrator.transform(iso_scores)             # type: ignore[union-attr]

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

    # ── Internal helpers ───────────────────────────────────────────────────

    def _fit_lgbm(self, x_tr, y_tr, x_val, y_val, w_tr=None):
        import pandas as pd
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        params = dict(self.lgbm_params)
        early = int(params.pop("early_stopping_rounds", 50) or 50)
        params.setdefault("random_state", self.random_seed)
        model = LGBMClassifier(**params)
        callbacks = [early_stopping(early, verbose=False), log_evaluation(period=0)]
        df_tr = pd.DataFrame(x_tr, columns=self.feature_names_)
        df_val = pd.DataFrame(x_val, columns=self.feature_names_)
        if len(set(y_val.tolist())) > 1:
            model.fit(df_tr, y_tr, eval_set=[(df_val, y_val)], callbacks=callbacks,
                      sample_weight=w_tr)
        else:
            model.fit(df_tr, y_tr, sample_weight=w_tr)
        return model

    def _predict_lgbm(self, model, x):
        import pandas as pd
        df = pd.DataFrame(x, columns=self.feature_names_) if self.feature_names_ else x
        proba = model.predict_proba(df)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _fit_catboost(self, x_tr, y_tr, x_val, y_val, w_tr=None):
        from catboost import CatBoostClassifier

        params = dict(self.catboost_params)
        params.setdefault("random_seed", self.random_seed)
        params.setdefault("verbose", 0)
        params.setdefault("eval_metric", "AUC")
        params.setdefault("early_stopping_rounds", 50)
        model = CatBoostClassifier(**params)
        if len(set(y_val.tolist())) > 1:
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False,
                      sample_weight=w_tr)
        else:
            model.fit(x_tr, y_tr, verbose=False, sample_weight=w_tr)
        return model

    def _predict_catboost(self, model, x):
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _fit_xgb(self, x_tr, y_tr, x_val, y_val, w_tr=None):
        from xgboost import XGBClassifier

        params = dict(self.xgb_params)
        params.setdefault("random_state", self.random_seed)
        params.setdefault("eval_metric", "auc")
        params.setdefault("verbosity", 0)
        early = int(params.pop("early_stopping_rounds", 50) or 50)
        params["early_stopping_rounds"] = early
        model = XGBClassifier(**params)
        if len(set(y_val.tolist())) > 1:
            model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False,
                      sample_weight=w_tr)
        else:
            model.fit(x_tr, y_tr, sample_weight=w_tr)
        return model

    def _predict_xgb(self, model, x):
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _fit_st(self, matrices_tr, y_tr, matrices_val, y_val, w_tr=None):
        try:
            from src.models.set_transformer import SetTransformerBot
            input_dim = matrices_tr[0].shape[1] if matrices_tr else 33
            params = dict(self.set_transformer_params)
            params.setdefault("input_dim", input_dim)
            params.setdefault("random_seed", self.random_seed)
            model = SetTransformerBot(**params)
            model.fit(matrices_tr, y_tr, matrices_val, y_val, sample_weight=w_tr)
            return model
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Set Transformer fold failed: %s", e)
            return None

    def _predict_st(self, model, matrices):
        try:
            return model.predict_proba(matrices)
        except Exception:
            return np.full(len(matrices), 0.5)
