from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Strategy3Context:
    """Mutable context shared across the Strategy 3 analysis pipeline."""

    repo_root: Path
    output_dir: Path
    rng_seed: int = 42
    version_info: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    discovered_files: List[Path] = field(default_factory=list)

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        print(entry)


class Strategy3Pipeline:
    """End-to-end pipeline orchestrating the Strategy 3 analysis."""

    TENOR_SET = (2, 5, 10, 20)

    def __init__(self, context: Strategy3Context) -> None:
        self.ctx = context
        self.panel_df: Optional[pd.DataFrame] = None
        self.tenor_liq: Optional[pd.DataFrame] = None
        self.arbitrage_panel: Optional[pd.DataFrame] = None
        self.panel_merged: Optional[pd.DataFrame] = None
        self.state_space_results: Optional[pd.DataFrame] = None
        self.state_space_summary: Optional[pd.DataFrame] = None
        self.msar_params: Optional[pd.DataFrame] = None
        self.msar_states: Optional[pd.DataFrame] = None
        self.half_life_ts: Optional[pd.DataFrame] = None
        self.regression_results: Optional[pd.DataFrame] = None
        self.robustness_summary: Optional[pd.DataFrame] = None
        self.concentration_metrics: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Environment and setup helpers
    # ------------------------------------------------------------------
    def setup_environment(self) -> None:
        import numpy
        import pandas

        np.random.seed(self.ctx.rng_seed)
        self.ctx.log(f"Random seed set to {self.ctx.rng_seed}")

        self.ctx.version_info = {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        }
        try:
            import statsmodels

            self.ctx.version_info["statsmodels"] = statsmodels.__version__
        except Exception as exc:  # pragma: no cover - defensive
            self.ctx.log(f"Unable to read statsmodels version: {exc}")
        try:
            import linearmodels

            self.ctx.version_info["linearmodels"] = linearmodels.__version__
        except Exception as exc:  # pragma: no cover - defensive
            self.ctx.log(f"Unable to read linearmodels version: {exc}")

        self.ctx.output_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.log(f"Output directory ready: {self.ctx.output_dir}")

        (self.ctx.output_dir / "logs").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading and validation
    # ------------------------------------------------------------------
    def load_crsp_panel(self, panel_path: Path) -> pd.DataFrame:
        self.ctx.log(f"Loading CRSP Treasury panel from {panel_path}")
        df = pd.read_parquet(panel_path)
        df["qdate"] = pd.to_datetime(df["qdate"])
        for col in ["matdt", "datdt"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        df["bid_ask_spread"] = df["ask"] - df["bid"]
        df = df.sort_values(["qdate", "crspid"]).reset_index(drop=True)
        self.panel_df = df
        self.ctx.log(f"Panel loaded: {len(df):,} rows")
        return df

    def assign_tenor_bucket(self, tenor_years: pd.Series) -> pd.Series:
        def nearest(value: float) -> Optional[int]:
            if pd.isna(value):
                return np.nan
            diffs = {tenor: abs(value - tenor) for tenor in self.TENOR_SET}
            tenor, delta = min(diffs.items(), key=lambda item: item[1])
            # guard against extremely long maturities (e.g., 30-year)
            if delta > 7:
                return np.nan
            return tenor

        return tenor_years.apply(nearest)

    def build_liquidity_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.panel_df is None:
            raise RuntimeError("Panel must be loaded before building liquidity tables")
        df = self.panel_df.copy()
        if "matdt" not in df.columns:
            raise ValueError("Panel missing maturity date column 'matdt'")
        df["tenor_years"] = (df["matdt"] - df["qdate"]).dt.days / 365.25
        df["tenor_bucket"] = self.assign_tenor_bucket(df["tenor_years"])
        df = df.dropna(subset=["tenor_bucket", "pubout"])
        df["tenor_bucket"] = df["tenor_bucket"].astype(int)
        self.ctx.log(
            "Tenor assignment completed for liquidity tables "
            f"({df['tenor_bucket'].nunique()} buckets)"
        )

        def compute_issue_concentration(group: pd.DataFrame, top_k: int) -> float:
            if group["pubout"].sum() <= 0:
                return np.nan
            shares = group["pubout"] / group["pubout"].sum()
            return shares.sort_values(ascending=False).head(top_k).sum()

        grouped = df.groupby(["qdate", "tenor_bucket"])
        tenor_liq = grouped.agg(
            bid_ask_spread=("bid_ask_spread", "median"),
            pubout=("pubout", "sum"),
            n_issues=("crspid", "nunique"),
        ).reset_index()
        hhi = grouped.apply(
            lambda g: np.nan
            if g["pubout"].sum() <= 0
            else ((g["pubout"] / g["pubout"].sum()) ** 2).sum()
        )
        tenor_liq["liq_hhi"] = hhi.values
        tenor_liq["issue_conc_top3"] = grouped.apply(lambda g: compute_issue_concentration(g, 3)).values
        tenor_liq["issue_conc_top5"] = grouped.apply(lambda g: compute_issue_concentration(g, 5)).values

        agg = tenor_liq.groupby("qdate").agg(
            bid_ask_spread=("bid_ask_spread", "mean"),
            pubout=("pubout", "sum"),
            n_issues=("n_issues", "sum"),
        ).reset_index()

        tenor_liq_path = self.ctx.repo_root / "tenor_liq.parquet"
        tenor_liq_csv_path = self.ctx.repo_root / "tenor_liq.csv"
        tenor_liq.to_parquet(tenor_liq_path, index=False)
        tenor_liq.to_csv(tenor_liq_csv_path, index=False)
        self.ctx.log(
            f"Saved tenor-level liquidity tables to {tenor_liq_path} and {tenor_liq_csv_path}"
        )

        agg_path = self.ctx.repo_root / "crsp_treasury_agg.csv"
        agg.to_csv(agg_path, index=False)
        self.ctx.log(f"Saved aggregate Treasury metrics to {agg_path}")

        self.tenor_liq = tenor_liq
        self.concentration_metrics = tenor_liq[[
            "qdate",
            "tenor_bucket",
            "pubout",
            "n_issues",
            "liq_hhi",
            "issue_conc_top3",
            "issue_conc_top5",
        ]].copy()
        return agg, tenor_liq

    # ------------------------------------------------------------------
    # Arbitrage discovery and harmonisation
    # ------------------------------------------------------------------
    def discover_arbitrage_files(self, patterns: Optional[Iterable[str]] = None) -> List[Path]:
        if patterns is None:
            patterns = [
                "arb_*.csv",
                "*arbitrage*.csv",
                "*basis*.csv",
                "*half*life*.csv",
                "*event*fit*.csv",
                "*half*life*.parquet",
            ]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(self.ctx.repo_root.rglob(pattern))
        files = sorted({path for path in files if path.is_file()})
        self.ctx.discovered_files = files
        self.ctx.log(f"Discovered {len(files)} arbitrage/half-life files")
        return files

    def _normalise_long_panel(self, df: pd.DataFrame, date_col: str, tenor_col: str, value_col: str, value_name: str) -> pd.DataFrame:
        panel = df[[date_col, tenor_col, value_col]].copy()
        panel = panel.rename(columns={date_col: "qdate", tenor_col: "tenor_bucket", value_col: value_name})
        panel["qdate"] = pd.to_datetime(panel["qdate"])
        panel["tenor_bucket"] = panel["tenor_bucket"].astype(int)
        return panel

    def _normalise_wide_panel(self, df: pd.DataFrame, date_col: str, prefix: str, value_name: str) -> pd.DataFrame:
        tenor_cols = [col for col in df.columns if col.startswith(prefix)]
        if not tenor_cols:
            raise ValueError("No tenor columns found for wide panel normalisation")
        id_vars = [date_col]
        panel = df[id_vars + tenor_cols].melt(id_vars=id_vars, var_name="tenor_label", value_name=value_name)
        panel["tenor_bucket"] = panel["tenor_label"].str.replace(prefix, "", regex=False).astype(int)
        panel = panel.drop(columns=["tenor_label"])
        panel = panel.rename(columns={date_col: "qdate"})
        panel["qdate"] = pd.to_datetime(panel["qdate"])
        return panel

    def build_arbitrage_panel(self, files: Optional[List[Path]] = None) -> pd.DataFrame:
        if files is None:
            files = self.ctx.discovered_files
        panels: List[pd.DataFrame] = []
        metadata: List[str] = []
        for path in files:
            try:
                if path.suffix.lower() == ".csv":
                    raw = pd.read_csv(path)
                elif path.suffix.lower() == ".parquet":
                    raw = pd.read_parquet(path)
                else:
                    continue
            except Exception as exc:
                self.ctx.log(f"Failed to load {path}: {exc}")
                continue

            lower_cols = {col.lower(): col for col in raw.columns}
            if {"date", "tenor", "basis"}.issubset(lower_cols.keys()):
                df = self._normalise_long_panel(
                    raw,
                    lower_cols["date"],
                    lower_cols["tenor"],
                    lower_cols["basis"],
                    "arb",
                )
                df["source_file"] = str(path.relative_to(self.ctx.repo_root))
                panels.append(df)
                metadata.append(f"long-basis:{path.name}")
                continue

            if {"valuation_date", "basis"}.issubset(lower_cols.keys()) and "tenor" not in lower_cols:
                df = raw.rename(columns={lower_cols["valuation_date"]: "qdate", lower_cols["basis"]: "arb"})
                df["qdate"] = pd.to_datetime(df["qdate"])
                df["tenor_bucket"] = 10
                df = df[["qdate", "tenor_bucket", "arb"]]
                df["source_file"] = str(path.relative_to(self.ctx.repo_root))
                panels.append(df)
                metadata.append(f"synthetic-basis:{path.name}")
                continue

            wide_cols = [col for col in raw.columns if col.lower().startswith("arb_")]
            if wide_cols and any(col.lower() == "qdate" or col.lower() == "date" for col in raw.columns):
                date_col = next(col for col in raw.columns if col.lower() in {"qdate", "date"})
                df = self._normalise_wide_panel(raw, date_col=date_col, prefix="arb_", value_name="arb")
                df["source_file"] = str(path.relative_to(self.ctx.repo_root))
                panels.append(df)
                metadata.append(f"wide-arb:{path.name}")
                continue

            if "half" in path.name.lower():
                # skip summary-only half-life tables here; handled later
                metadata.append(f"halflife-metadata:{path.name}")
                continue

            metadata.append(f"unparsed:{path.name}")

        if not panels:
            raise RuntimeError("No arbitrage data sources parsed successfully")

        arb_panel = pd.concat(panels, ignore_index=True)
        arb_panel = arb_panel.sort_values(["qdate", "tenor_bucket"]).drop_duplicates(
            subset=["qdate", "tenor_bucket", "source_file"], keep="last"
        )
        self.ctx.log(
            "Arbitrage panel built from sources: " + ", ".join(sorted(set(metadata)))
        )

        # combine multiple sources by averaging when duplicates exist
        arb_panel = (
            arb_panel.groupby(["qdate", "tenor_bucket"], as_index=False)
            .agg(arb=("arb", "mean"))
        )
        self.arbitrage_panel = arb_panel
        return arb_panel

    # ------------------------------------------------------------------
    # Canonical panel construction
    # ------------------------------------------------------------------
    def construct_canonical_panel(self) -> pd.DataFrame:
        if self.arbitrage_panel is None:
            raise RuntimeError("Arbitrage panel not built")
        if self.tenor_liq is None:
            raise RuntimeError("Liquidity tables not prepared")

        tenor_liq = self.tenor_liq.copy()
        tenor_liq["qdate"] = pd.to_datetime(tenor_liq["qdate"])
        panel = pd.merge(
            self.arbitrage_panel,
            tenor_liq,
            on=["qdate", "tenor_bucket"],
            how="left",
        )

        fill_cols = ["bid_ask_spread", "pubout", "n_issues", "liq_hhi", "issue_conc_top3", "issue_conc_top5"]
        for col in fill_cols:
            if col in panel.columns:
                panel[col] = panel.groupby("tenor_bucket")[col].transform(lambda s: s.ffill(limit=90).bfill(limit=90))

        panel = panel.sort_values(["tenor_bucket", "qdate"])
        panel["arb"] = panel.groupby("tenor_bucket")["arb"].transform(
            lambda s: s.ffill(limit=5)
        )

        # detrended arbitrage via HP filter fallback to rolling mean if statsmodels missing
        try:
            import statsmodels.api as sm

            detrended_list = []
            for tenor, group in panel.groupby("tenor_bucket"):
                if len(group) < 30:
                    detrended_list.append(pd.Series(np.nan, index=group.index))
                    continue
                cycle, trend = sm.tsa.filters.hpfilter(group["arb"], lamb=129600)
                detrended_list.append(cycle)
            panel["m"] = pd.concat(detrended_list).sort_index()
        except Exception as exc:
            self.ctx.log(f"HP filter failed ({exc}); falling back to rolling mean detrend")
            panel["m"] = panel.groupby("tenor_bucket")["arb"].transform(
                lambda s: s - s.rolling(window=60, min_periods=20, center=False).mean()
            )

        # compute liquidity z-score by tenor/month
        panel["month"] = panel["qdate"].dt.to_period("M").dt.to_timestamp()
        panel["liquidity_z"] = (
            panel.groupby(["tenor_bucket", "month"])["bid_ask_spread"]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) not in (0, np.nan) else 0.0)
        )
        panel["issue_conc_top3"] = panel.get("issue_conc_top3")
        panel["issue_conc_top5"] = panel.get("issue_conc_top5")

        # create canonical concentration metrics for export
        self.concentration_metrics = panel[[
            "qdate",
            "tenor_bucket",
            "liq_hhi",
            "issue_conc_top3",
            "issue_conc_top5",
            "pubout",
            "n_issues",
        ]].drop_duplicates().sort_values(["tenor_bucket", "qdate"])

        self.panel_merged = panel
        return panel

    # ------------------------------------------------------------------
    # State-space modelling
    # ------------------------------------------------------------------
    def run_state_space_models(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.panel_merged is None:
            raise RuntimeError("Canonical panel must be constructed before SSM")
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        records = []
        summaries = []
        for tenor, group in self.panel_merged.groupby("tenor_bucket"):
            series = group[["qdate", "arb"]].dropna()
            if len(series) < 50:
                self.ctx.log(f"Skipping tenor {tenor}: insufficient data for SSM")
                continue
            y = series.set_index("qdate")["arb"]
            model = UnobservedComponents(endog=y, level="local level", autoregressive=1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(disp=False)
            except Exception as exc:
                self.ctx.log(f"SSM fit failed for tenor {tenor}: {exc}")
                continue

            smoothed = res.smoothed_state.T
            filtered = res.filtered_state.T
            state_names = res.model.state_names
            states_df = pd.DataFrame(
                {
                    "qdate": y.index,
                    "tenor_bucket": tenor,
                }
            )
            for idx, name in enumerate(state_names):
                states_df[f"smoothed_{name}"] = smoothed[:, idx]
                states_df[f"filtered_{name}"] = filtered[:, idx]
            phi_param = res.params.get("ar.L1", res.params.get("autoregressive.ar.L1", np.nan))
            sigma2_level = float(np.squeeze(res.filter_results.obs_cov)) if hasattr(res.filter_results, "obs_cov") else np.nan
            states_df["phi"] = float(phi_param) if phi_param is not None and not np.isnan(phi_param) else np.nan
            states_df["sigma2_level"] = sigma2_level
            states_df["loglike"] = float(res.llf)
            records.append(states_df)

            phi = float(phi_param) if phi_param is not None and not np.isnan(phi_param) else np.nan
            phi_se = (res.bse.get("ar.L1") if hasattr(res, "bse") and "ar.L1" in res.bse else res.bse.get("autoregressive.ar.L1", np.nan) if hasattr(res, "bse") else np.nan)
            hl = self._half_life_from_phi(phi)
            hl_ci = self._half_life_ci(phi, phi_se)
            summaries.append(
                {
                    "tenor_bucket": tenor,
                    "phi": phi,
                    "phi_se": phi_se,
                    "hl_disturbance": hl,
                    "hl_disturbance_lower": hl_ci[0],
                    "hl_disturbance_upper": hl_ci[1],
                    "loglike": res.llf,
                    "aic": res.aic,
                    "bic": res.bic,
                }
            )

        if not records:
            raise RuntimeError("No state-space models succeeded")

        states = pd.concat(records, ignore_index=True)
        summary_df = pd.DataFrame(summaries)
        self.state_space_results = states
        self.state_space_summary = summary_df
        return states, summary_df

    @staticmethod
    def _half_life_from_phi(phi: float) -> float:
        if phi is None or np.isnan(phi) or abs(phi) >= 1 or abs(phi) < 1e-6:
            return np.nan
        return -np.log(2) / np.log(abs(phi))

    @staticmethod
    def _half_life_ci(phi: float, phi_se: float, confidence: float = 0.95) -> Tuple[float, float]:
        if phi is None or np.isnan(phi) or phi_se is None or np.isnan(phi_se) or phi_se <= 0:
            return (np.nan, np.nan)
        z = 1.959963984540054
        lower_phi = phi - z * phi_se
        upper_phi = phi + z * phi_se
        return (
            Strategy3Pipeline._half_life_from_phi(upper_phi),
            Strategy3Pipeline._half_life_from_phi(lower_phi),
        )

    # ------------------------------------------------------------------
    # Markov-switching AR models
    # ------------------------------------------------------------------
    def run_msar_models(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.state_space_results is None:
            raise RuntimeError("Run state-space models before MS-AR")
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        msar_params_records: List[Dict[str, float]] = []
        msar_states_records: List[pd.DataFrame] = []
        for tenor, group in self.state_space_results.groupby("tenor_bucket"):
            if "smoothed_autoregressive" not in group.columns:
                continue
            epsilon = group[["qdate", "smoothed_autoregressive"]].dropna()
            epsilon = epsilon.set_index("qdate").sort_index()["smoothed_autoregressive"]
            if len(epsilon) < 80:
                self.ctx.log(f"Skipping MS-AR for tenor {tenor}: insufficient data")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = MarkovAutoregression(epsilon, k_regimes=2, order=1, switching_variance=True)
                    res = mod.fit(disp=False)
            except Exception as exc:
                self.ctx.log(f"MS-AR fit failed for tenor {tenor}: {exc}")
                continue

            regime_params = res.params.reshape((mod.k_regimes, -1))
            phi_values = res.params[mod.parameters["ar"]]
            if np.isscalar(phi_values):
                phi_values = np.repeat(phi_values, mod.k_regimes)
            for regime in range(mod.k_regimes):
                phi = phi_values[regime]
                hl = self._half_life_from_phi(phi)
                msar_params_records.append(
                    {
                        "tenor_bucket": tenor,
                        "regime": regime,
                        "phi": phi,
                        "half_life": hl,
                        "variance": res.params[mod.parameters["sigma2"]][regime],
                    }
                )

            transition = res.transition_matrix
            for i in range(transition.shape[0]):
                for j in range(transition.shape[1]):
                    msar_params_records.append(
                        {
                            "tenor_bucket": tenor,
                            "regime_from": i,
                            "regime_to": j,
                            "transition_prob": transition[i, j],
                        }
                    )

            smoothed_probs = res.smoothed_marginal_probabilities
            probs_df = pd.DataFrame(
                {
                    "qdate": epsilon.index,
                    **{f"prob_regime_{i}": smoothed_probs[i] for i in range(mod.k_regimes)},
                }
            )
            probs_df["tenor_bucket"] = tenor
            probs_df["expected_half_life"] = sum(
                self._half_life_from_phi(phi_values[i]) * probs_df[f"prob_regime_{i}"]
                for i in range(mod.k_regimes)
            )
            msar_states_records.append(probs_df)

        if not msar_params_records:
            self.ctx.log("No MS-AR models succeeded; creating empty placeholders")
            params_df = pd.DataFrame(columns=["tenor_bucket", "regime", "phi", "half_life", "variance", "regime_from", "regime_to", "transition_prob"])
            states_df = pd.DataFrame(columns=["qdate", "tenor_bucket", "prob_regime_0", "prob_regime_1", "expected_half_life"])
            self.msar_params = params_df
            self.msar_states = states_df
            return params_df, states_df

        params_df = pd.DataFrame(msar_params_records)
        states_df = pd.concat(msar_states_records, ignore_index=True)
        self.msar_params = params_df
        self.msar_states = states_df
        return params_df, states_df

    # ------------------------------------------------------------------
    # Half-life time series construction
    # ------------------------------------------------------------------
    def build_half_life_timeseries(self) -> pd.DataFrame:
        if self.panel_merged is None or self.state_space_results is None:
            raise RuntimeError("Run panel merge and SSM before half-life extraction")

        # Rolling AR(1) on smoothed disturbances to obtain time-varying half-life
        def rolling_half_life(series: pd.Series, window: int = 126, min_periods: int = 63) -> pd.Series:
            values = []
            index = series.index
            for end in range(len(series)):
                start = max(0, end - window + 1)
                if end - start + 1 < min_periods:
                    values.append(np.nan)
                    continue
                window_series = series.iloc[start : end + 1]
                if window_series.isna().sum() > 0:
                    values.append(np.nan)
                    continue
                model = np.polyfit(window_series[:-1], window_series[1:], deg=1)
                phi = model[0]
                values.append(self._half_life_from_phi(phi))
            return pd.Series(values, index=index)

        hl_records = []
        for tenor, group in self.state_space_results.groupby("tenor_bucket"):
            col = None
            for candidate in ["smoothed_autoregressive", "smoothed_ar.L1", "smoothed_irregular"]:
                if candidate in group.columns:
                    col = candidate
                    break
            if col is None:
                continue
            group = group.sort_values("qdate").set_index("qdate")
            series = group[col].dropna()
            if series.empty:
                continue
            hl_series = rolling_half_life(series)
            hl_series.name = "hl_ssm"
            df = hl_series.reset_index().rename(columns={"index": "qdate"})
            df["tenor_bucket"] = tenor
            hl_records.append(df)

        if not hl_records:
            hl_df = pd.DataFrame(columns=["qdate", "tenor_bucket", "hl_ssm"])
        else:
            hl_df = pd.concat(hl_records, ignore_index=True)

        if self.msar_states is not None and not self.msar_states.empty:
            msar = self.msar_states.copy()
            if "expected_half_life" in msar.columns:
                msar = msar.rename(columns={"expected_half_life": "hl_msar"})
                hl_df = pd.merge(hl_df, msar[["qdate", "tenor_bucket", "hl_msar"]], on=["qdate", "tenor_bucket"], how="left")

        self.half_life_ts = hl_df
        return hl_df

    # ------------------------------------------------------------------
    # Panel regressions
    # ------------------------------------------------------------------
    def run_panel_regressions(self) -> pd.DataFrame:
        if self.half_life_ts is None or self.panel_merged is None:
            raise RuntimeError("Half-life time series and panel must be ready")
        import statsmodels.api as sm
        from statsmodels.regression.quantile_regression import QuantReg

        panel = pd.merge(
            self.half_life_ts,
            self.panel_merged,
            on=["qdate", "tenor_bucket"],
            how="left",
        )
        panel = panel.dropna(subset=["hl_ssm", "bid_ask_spread", "pubout", "liq_hhi"])
        panel = panel.sort_values(["tenor_bucket", "qdate"])
        panel["month"] = panel["qdate"].dt.to_period("M").dt.to_timestamp()

        regressors = [
            "bid_ask_spread",
            "pubout",
            "n_issues",
            "liq_hhi",
            "issue_conc_top3",
            "liquidity_z",
        ]
        for col in regressors:
            panel[f"L1_{col}"] = panel.groupby("tenor_bucket")[col].shift(1)
        panel = panel.dropna(subset=[f"L1_{col}" for col in regressors])

        panel_reset = panel.reset_index(drop=True)
        panel_reset["month_str"] = panel_reset["month"].dt.strftime("%Y-%m")
        X_base = panel_reset[[f"L1_{col}" for col in regressors]]
        dummies_entity = pd.get_dummies(panel_reset["tenor_bucket"].astype(str), prefix="tenor", drop_first=True)
        dummies_time = pd.get_dummies(panel_reset["month_str"], prefix="month", drop_first=True)
        X_df = pd.concat([X_base, dummies_entity, dummies_time], axis=1)
        X_df = sm.add_constant(X_df, has_constant="add")
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_series = panel_reset["hl_ssm"].astype(float)

        clusters = panel_reset["tenor_bucket"].astype(str) + "_" + panel_reset["month_str"]
        X_np = X_df.to_numpy(dtype=float)
        y_np = y_series.to_numpy(dtype=float)
        fe_model = sm.OLS(y_np, X_np)
        fe_res = fe_model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
        fe_res.model.data.xnames = list(X_df.columns)
        fe_res.model.data.ynames = ['hl_ssm']

        panel_reset["resid"] = fe_res.resid
        phi_by_tenor: Dict[int, float] = {}
        for tenor, grp in panel_reset.groupby("tenor_bucket"):
            grp = grp.sort_values("qdate")
            if grp["resid"].count() < 10:
                continue
            resid_vals = grp["resid"].values
            phi = np.polyfit(resid_vals[:-1], resid_vals[1:], deg=1)[0]
            phi_by_tenor[int(tenor)] = float(np.clip(phi, -0.99, 0.99))

        transformed_rows = []
        transformed_targets = []
        last_index_by_tenor: Dict[int, int] = {}
        for idx, row in panel_reset.iterrows():
            tenor = int(row["tenor_bucket"])
            phi = phi_by_tenor.get(tenor, 0.0)
            prev_idx = last_index_by_tenor.get(tenor)
            x_row = X_df.iloc[idx].values.astype(float)
            y_val = float(y_series.iloc[idx])
            if prev_idx is not None:
                x_row = x_row - phi * X_df.iloc[prev_idx].values.astype(float)
                y_val = y_val - phi * float(y_series.iloc[prev_idx])
            transformed_rows.append(x_row)
            transformed_targets.append(y_val)
            last_index_by_tenor[tenor] = idx

        X_gls = np.vstack(transformed_rows)
        y_gls = np.array(transformed_targets)
        gls_model = sm.OLS(y_gls, X_gls)
        gls_res = gls_model.fit(cov_type="HC1")

        demeaned = panel_reset.copy()
        demeaned["hl_ssm_demeaned"] = demeaned.groupby("tenor_bucket")["hl_ssm"].transform(lambda s: s - s.mean())
        demean_cols = [f"L1_{col}" for col in regressors]
        for col in demean_cols:
            demeaned[f"{col}_demeaned"] = demeaned.groupby("tenor_bucket")[col].transform(lambda s: s - s.mean())
        required_cols = ["hl_ssm_demeaned"] + [f"{col}_demeaned" for col in demean_cols]
        dq = demeaned.dropna(subset=required_cols)
        y_q = dq["hl_ssm_demeaned"]
        X_q = dq[[f"{col}_demeaned" for col in demean_cols]]
        quant_model = QuantReg(y_q, X_q)
        quant_res = quant_model.fit(q=0.5)

        results = []
        for label, res in [
            ("FE-OLS", fe_res),
            ("GLS-Parks", gls_res),
        ]:
            params = res.params
            ses = res.bse
            if hasattr(params, 'index'):
                param_series = params
            else:
                names = getattr(res.model, 'data', None)
                if names is not None and hasattr(names, 'xnames'):
                    param_series = pd.Series(params, index=names.xnames)
                else:
                    param_series = pd.Series(params, index=[f'param_{i}' for i in range(len(params))])
            if hasattr(ses, 'index'):
                se_series = ses
            else:
                se_series = pd.Series(ses, index=param_series.index)
            if hasattr(res, 'pvalues'):
                raw_pvals = res.pvalues
                if hasattr(raw_pvals, 'index'):
                    pvals_series = raw_pvals
                else:
                    pvals_series = pd.Series(raw_pvals, index=param_series.index)
            else:
                pvals_series = pd.Series(np.nan, index=param_series.index)
            for name in param_series.index:
                se_val = se_series[name]
                estimate = param_series[name]
                denom = se_val if se_val not in (0, np.nan) else np.nan
                t_stat = estimate / denom if denom not in (0, np.nan) else np.nan
                results.append(
                    {
                        "model": label,
                        "term": name,
                        "estimate": estimate,
                        "std_error": se_val,
                        "t_stat": t_stat,
                        "p_value": pvals_series.get(name, np.nan),
                    }
                )
            results.append(
                {
                    "model": label,
                    "term": "_fit_stats",
                    "estimate": res.rsquared if hasattr(res, "rsquared") else np.nan,
                    "std_error": res.ssr if hasattr(res, "ssr") else np.nan,
                    "t_stat": res.aic if hasattr(res, "aic") else np.nan,
                    "p_value": res.bic if hasattr(res, "bic") else np.nan,
                }
            )

        if hasattr(quant_res, 'params'):
            for name, param in quant_res.params.items():
                se = quant_res.bse.get(name, np.nan) if hasattr(quant_res, "bse") else np.nan
                results.append(
                    {
                        "model": "Quantile-0.5",
                        "term": name,
                        "estimate": param,
                        "std_error": se,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                    }
                )

        results_df = pd.DataFrame(results)
        self.regression_results = results_df
        return results_df

    # ------------------------------------------------------------------
    # Robustness and forecasts
    # ------------------------------------------------------------------
    def run_robustness(self) -> pd.DataFrame:
        if self.panel_merged is None or self.state_space_results is None:
            raise RuntimeError("Panel and state-space results required for robustness checks")
        from statsmodels.tsa.ar_model import AutoReg

        metrics = []
        for tenor, group in self.panel_merged.groupby("tenor_bucket"):
            series = group.sort_values("qdate").set_index("qdate")["arb"].dropna()
            if len(series) < 200:
                continue
            ar_mod = AutoReg(series, lags=1, old_names=False).fit()
            ar_pred = ar_mod.predict(start=1, end=len(series) - 1)
            ar_actual = series.iloc[1:]
            ar_errors = ar_actual - ar_pred

            ssm_group = self.state_space_results[self.state_space_results["tenor_bucket"] == tenor]
            ssm_group = ssm_group.set_index("qdate").reindex(series.index)
            ssm_forecast = ssm_group["filtered_level"].shift(1)
            ssm_errors = series - ssm_forecast

            def error_stats(errors: pd.Series) -> Dict[str, float]:
                errors = errors.dropna()
                return {
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "mae": float(np.mean(np.abs(errors))),
                }

            ar_stats = error_stats(ar_errors)
            ssm_stats = error_stats(ssm_errors)
            metrics.append(
                {
                    "tenor_bucket": tenor,
                    "model": "AR(1)",
                    **ar_stats,
                }
            )
            metrics.append(
                {
                    "tenor_bucket": tenor,
                    "model": "StateSpace",
                    **ssm_stats,
                }
            )

            # Toy PnL: sign of arb * next change minus half spread cost
            spread = group.set_index("qdate")["bid_ask_spread"].reindex(series.index)
            pnl = -np.sign(series) * series.diff(-1).fillna(0) - spread.fillna(spread.median()) * 0.5
            metrics.append(
                {
                    "tenor_bucket": tenor,
                    "model": "ToyPnL",
                    "rmse": float(pnl.mean()),
                    "mae": float(pnl.std()),
                }
            )

        robustness_df = pd.DataFrame(metrics)
        self.robustness_summary = robustness_df
        return robustness_df

    # ------------------------------------------------------------------
    # Utility: save outputs
    # ------------------------------------------------------------------
    def save_outputs(self) -> None:
        if self.panel_merged is not None:
            panel_path = self.ctx.output_dir / "panel_merged.parquet"
            self.panel_merged.to_parquet(panel_path, index=False)
            self.ctx.log(f"Saved merged panel to {panel_path}")
        if self.state_space_results is not None:
            ssm_path = self.ctx.output_dir / "state_space_estimates.csv"
            self.state_space_results.to_csv(ssm_path, index=False)
            self.ctx.log(f"Saved state-space states to {ssm_path}")
        if self.msar_params is not None:
            msar_path = self.ctx.output_dir / "msar_params.csv"
            self.msar_params.to_csv(msar_path, index=False)
            self.ctx.log(f"Saved MS-AR parameters to {msar_path}")
        if self.half_life_ts is not None:
            hl_path = self.ctx.output_dir / "half_life_estimates.csv"
            self.half_life_ts.to_csv(hl_path, index=False)
            self.ctx.log(f"Saved half-life estimates to {hl_path}")
        if self.regression_results is not None:
            reg_path = self.ctx.output_dir / "panel_regression_results.csv"
            self.regression_results.to_csv(reg_path, index=False)
            self.ctx.log(f"Saved regression results to {reg_path}")
        if self.concentration_metrics is not None:
            conc_path = self.ctx.output_dir / "concentration_metrics.csv"
            self.concentration_metrics.to_csv(conc_path, index=False)
            self.ctx.log(f"Saved concentration metrics to {conc_path}")
        if self.robustness_summary is not None:
            rob_path = self.ctx.output_dir / "robustness_summary.csv"
            self.robustness_summary.to_csv(rob_path, index=False)
            self.ctx.log(f"Saved robustness metrics to {rob_path}")

    def export_logs(self) -> None:
        log_path = self.ctx.output_dir / "logs" / "pipeline_log.json"
        log_data = {
            "version_info": self.ctx.version_info,
            "entries": self.ctx.logs,
            "discovered_files": [str(p.relative_to(self.ctx.repo_root)) for p in self.ctx.discovered_files],
        }
        log_path.write_text(json.dumps(log_data, indent=2))
        self.ctx.log(f"Pipeline log saved to {log_path}")
