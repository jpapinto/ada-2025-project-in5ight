import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def p25_p75_span(x: pd.Series) -> tuple[float, float, float]:
    lo, hi = x.quantile([0.25, 0.75])
    return float(lo), float(hi), float(hi - lo)


def effect_from_coefficient(beta: float, se: float, span: float, z: float = 1.96):
    # Effect in r units and a simple Wald CI based on beta +- z*se
    delta_r = span * beta
    low_r   = span * (beta - z * se)
    high_r  = span * (beta + z * se)
    return float(delta_r), float(low_r), float(high_r)


def r_to_pct(delta_r: float) -> float:
    # Your transform: %ΔE = (10^(Δr) - 1) * 100
    return (10.0**delta_r - 1.0) * 100.0


def run_within_tier_wls(
    d: pd.DataFrame,
    tier_name: str,
    pace_col: str,
    cons_col: str,
    *,
    r_col: str = "residual",
    duration_col: str = "duration_median",
    category_col: str = "category_cc",
    age_col: str = "channel_age_days",
    w_primary: str = "video_count_processed",
    w_fallback: str = "video_count_from_pace",
    print_summary: bool = False,
) -> tuple:
    """
    Fits a weighted least squares regression within a tier:
        r ~ pace + cons + log_duration + C(category) + log_age

    Notes:
        - Uses HC3 robust standard errors.
        - Weights prefer w_primary if present and has any non-null values, otherwise falls back
          to w_fallback; weights are clipped to be at least 1.
        - Adds 'log_duration' as log(duration_median) for duration_median > 0, else NaN.
        - Adds 'log_age' as log(channel_age_days + 1).
        - Drops rows with NaNs (or inf) in required regression columns.

        Effect calculation:
            For each key variable (pace, cons, log_duration), compute the p25 to p75 span within
            this tier and translate the coefficient into an effect in r-units:
                Δr = span * beta
            Then compute a Wald-style 95% CI in r-units using beta ± 1.96 * se.

        Percent transform:
            Effects are reported as %ΔE using:
                %ΔE = (10^(Δr) - 1) * 100

        Internal helper definitions used here:
            p25_p75_span(x):
                Returns (p25, p75, span) where span = p75 - p25.
            r_to_pct(delta_r):
                Applies %ΔE = (10^(Δr) - 1) * 100.

    Args:
        d (pd.DataFrame): Tier-specific input data.
        tier_name (str): Label for the tier (e.g., 'small', 'mid') used in outputs and printing.
        pace_col (str): Column name for posting pace predictor.
        cons_col (str): Column name for posting consistency predictor.
        r_col (str): Column name for the dependent variable. Defaults to 'residual'.
        duration_col (str): Column name for duration (seconds) used to build log_duration.
            Defaults to 'duration_median'.
        category_col (str): Column name for categorical control (wrapped with C()).
            Defaults to 'category_cc'.
        age_col (str): Column name for channel age in days used to build log_age.
            Defaults to 'channel_age_days'.
        w_primary (str): Preferred weight column. Defaults to 'video_count_processed'.
        w_fallback (str): Fallback weight column if primary is unavailable. Defaults to
            'video_count_from_pace'.
        print_summary (bool): If True, prints the statsmodels summary. Defaults to False.

    Returns:
        tuple: (model, effects_df, coefs_df)
            - model: statsmodels fitted WLS result (or None if no usable rows).
            - effects_df (pd.DataFrame): Per-variable p25->p75 effect table including CI in %ΔE.
              Empty if no usable rows.
            - coefs_df (pd.DataFrame): Coefficient table for pace, cons, and log_duration with
              beta, se, and pval. Empty if no usable rows.
    """
    d = d.copy().replace([np.inf, -np.inf], np.nan)

    d["log_duration"] = np.log(pd.to_numeric(d[duration_col], errors="coerce").where(d[duration_col] > 0))
    d["log_age"] = np.log(pd.to_numeric(d[age_col], errors="coerce") + 1.0)

    needed = [r_col, pace_col, cons_col, "log_duration", category_col, "log_age"]
    d = d.dropna(subset=needed)
    if d.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    # weights
    if (w_primary in d.columns) and d[w_primary].notna().any():
        d["wts"] = pd.to_numeric(d[w_primary], errors="coerce").fillna(1).clip(lower=1)
    elif w_fallback in d.columns:
        d["wts"] = pd.to_numeric(d[w_fallback], errors="coerce").fillna(1).clip(lower=1)
    else:
        d["wts"] = 1.0

    formula = f"{r_col} ~ {pace_col} + {cons_col} + log_duration + C({category_col}) + log_age"
    model = smf.wls(formula, data=d, weights=d["wts"]).fit(cov_type="HC3")

    if print_summary:
        print(f"\n=== {tier_name.upper()} | {pace_col} + {cons_col} | WLS (HC3) ===")
        print(model.summary())

    # coefficients for key vars
    key_vars = [pace_col, cons_col, "log_duration"]
    coefs_df = pd.DataFrame({
        "tier": tier_name,
        "spec": f"{pace_col}+{cons_col}",
        "var": key_vars,
        "beta": [float(model.params[v]) for v in key_vars],
        "se":   [float(model.bse[v]) for v in key_vars],
        "pval": [float(model.pvalues[v]) for v in key_vars],
    })

    # p25 to p75 effects (within tier)
    rows = []
    for v in key_vars:
        p25, p75, span = p25_p75_span(d[v])
        beta = float(model.params[v])
        se = float(model.bse[v])

        delta_r = span * beta
        lo_r = span * (beta - 1.96 * se)
        hi_r = span * (beta + 1.96 * se)

        rows.append({
            "tier": tier_name,
            "spec": f"{pace_col}+{cons_col}",
            "var": v,
            "p25": p25, "p75": p75, "span": span,
            "%ΔE_p25->p75": r_to_pct(delta_r),
            "%ΔE_CI_low": r_to_pct(lo_r),
            "%ΔE_CI_high": r_to_pct(hi_r),
        })

    effects_df = pd.DataFrame(rows)
    return model, effects_df, coefs_df


def run_spec_small_mid(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    *,
    print_summary: bool = False,
):
    """
    Runs the within-tier WLS spec for both Small and Mid tiers and aggregates results.

    Notes:
        - Calls run_within_tier_wls for each tier and collects:
          fitted models, p25->p75 effects, and coefficient tables.
        - If a tier has no usable rows after filtering, its model will be None and it will not
          contribute rows to the concatenated effects/coef outputs.

    Args:
        small_df (pd.DataFrame): Input data for the Small tier.
        mid_df (pd.DataFrame): Input data for the Mid tier.
        pace_col (str): Column name for posting pace predictor.
        cons_col (str): Column name for posting consistency predictor.
        print_summary (bool): If True, prints each tier's statsmodels summary.
            Defaults to False.

    Returns:
        tuple: (models, effects_df, coefs_df)
            - models (dict): Mapping tier_name -> fitted model (or None).
            - effects_df (pd.DataFrame): Concatenated per-tier effects tables.
            - coefs_df (pd.DataFrame): Concatenated per-tier coefficient tables.
    """
    models = {}
    effects = []
    coefs = []

    for tier_name, d in [("small", small_df), ("mid", mid_df)]:
        model, eff_df, coef_df = run_within_tier_wls(
            d=d,
            tier_name=tier_name,
            pace_col=pace_col,
            cons_col=cons_col,
            print_summary=print_summary,
        )
        models[tier_name] = model
        if not eff_df.empty:
            effects.append(eff_df)
        if not coef_df.empty:
            coefs.append(coef_df)

    effects_df = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    coefs_df = pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()
    return models, effects_df, coefs_df
