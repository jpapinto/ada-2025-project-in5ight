"""
Plotly utilities for:
- VIF tables (small vs mid)
- correlation heatmaps (absolute)
- 3x2 "recipe map" heatmaps with low-N fade
- effects bars with CIs
- coefficient bars with HC3 CIs and significance stars

Notes:
- This file uses a "washed Viridis" colorscale to reduce vividness.
- Duplicated helpers/functions from earlier drafts were removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.cm as cm


# -------------------------
# Styling helpers
# -------------------------

PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
)

BAR_PALETTE = ["#332288", "#CC6677"]


def mpl_to_plotly_colorscale(cmap_name: str = "coolwarm", n: int = 256):
    """
    Convert a Matplotlib colormap to a Plotly colorscale.
    Returns: [[0.0, "rgb(...)"], ..., [1.0, "rgb(...)"]]
    """
    cmap = cm.get_cmap(cmap_name, n)
    scale = []
    for i in range(n):
        r, g, b, _a = cmap(i)
        scale.append([i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return scale


def softened_viridis(mix_with_white: float = 0.35):
    """
    Viridis colorscale blended toward white.
    mix_with_white:
      0.0 -> original Viridis
      1.0 -> all white
    """
    base = pc.get_colorscale("Viridis")
    out = []
    for p, col in base:
        if col.startswith("#"):
            r, g, b = pc.hex_to_rgb(col)
        else:
            r, g, b = pc.unlabel_rgb(col)
        r2 = int(round(r + (255 - r) * mix_with_white))
        g2 = int(round(g + (255 - g) * mix_with_white))
        b2 = int(round(b + (255 - b) * mix_with_white))
        out.append([p, f"rgb({r2},{g2},{b2})"])
    return out


# -------------------------
# Naming / feature prep
# -------------------------

def pretty_name(v: str) -> str:
    if v == "log_duration":
        return "log(duration)"
    if v.startswith("pace") and v.endswith("_median"):
        w = v.replace("pace", "").replace("_median", "")
        return f"pace ({w}w)"
    if v.startswith("cons") and v.endswith("_median"):
        w = v.replace("cons", "").replace("_median", "")
        return f"consistency ({w}w)"
    return v


def _prep_struct_cols(
    d: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
):
    d = d.copy().replace([np.inf, -np.inf], np.nan)

    duration = pd.to_numeric(d[duration_col], errors="coerce")
    d["log_duration"] = np.log(duration.where(duration > 0))

    cols = [pace_col, cons_col, "log_duration"]
    X = d[cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return X, cols


# -------------------------
# VIF
# -------------------------

def _compute_vif_series(X: pd.DataFrame) -> pd.Series:
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty or len(X) < 5:
        return pd.Series({c: np.nan for c in X.columns})

    Xv = X.assign(const=1.0)
    vifs: dict[str, float] = {}
    for i, c in enumerate(Xv.columns):
        if c == "const":
            continue
        vifs[c] = float(variance_inflation_factor(Xv.values, i))
    return pd.Series(vifs)


def vif_table_small_mid(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
) -> pd.DataFrame:
    rows = []
    for tier, d in [("small", small_df), ("mid", mid_df)]:
        X, _cols = _prep_struct_cols(d, pace_col, cons_col, duration_col)
        rows.append(pd.Series(_compute_vif_series(X), name=tier))

    out = pd.DataFrame(rows)
    out = out.rename(columns={c: pretty_name(c) for c in out.columns})
    return out


# -------------------------
# Correlation heatmaps
# -------------------------

def plot_corr_small_mid(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pace_col: str,
    cons_col: str,
    duration_col: str = "duration_median",
    show: bool = False,
    viridis_wash: float = 0.35,
):
    Xs, cols = _prep_struct_cols(small_df, pace_col, cons_col, duration_col)
    Xm, _ = _prep_struct_cols(mid_df, pace_col, cons_col, duration_col)

    corr_s = Xs.corr().abs()
    corr_m = Xm.corr().abs()

    labels = [pretty_name(c) for c in cols]
    colorscale = softened_viridis(viridis_wash)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Small Tier", "Mid Tier"),
        horizontal_spacing=0.12,
    )

    zmin, zmax = 0.0, 1.0

    fig.add_trace(
        go.Heatmap(
            z=corr_s.values,
            x=labels,
            y=labels,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title="|corr|", len=0.8, y=0.5),
            text=np.round(corr_s.values, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="x=%{x}<br>y=%{y}<br>|corr|=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=corr_m.values,
            x=labels,
            y=labels,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            showscale=False,
            text=np.round(corr_m.values, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="x=%{x}<br>y=%{y}<br>|corr|=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Make y-axis go top to bottom so diagonal reads naturally
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        title="Correlations (absolute) by tier",
        height=420,
        width=950,
        margin=dict(l=50, r=30, t=60, b=40),
        **PLOTLY_THEME,
    )

    if show:
        fig.show()
    return fig


# -------------------------
# Recipe map binning helpers
# -------------------------

def _bin_edges_global(
    s: pd.Series,
    bins: int = 8,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.array([0.0, 1.0])

    lo = float(s.quantile(q_low))
    hi = float(s.quantile(q_high))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(s.min())
        hi = float(s.max())
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            return np.array([0.0, 1.0])

    edges = np.unique(np.linspace(lo, hi, bins + 1))
    if len(edges) < 3:
        edges = np.unique(np.linspace(float(s.min()), float(s.max()), bins + 1))
    return edges


def _centers_and_ranges(edges: np.ndarray):
    centers = (edges[:-1] + edges[1:]) / 2.0
    ranges = list(zip(edges[:-1], edges[1:]))
    return centers, ranges


def _full_categories_from_edges(edges: np.ndarray):
    centers, _ = _centers_and_ranges(edges)
    return (
        pd.cut(pd.Series(centers), bins=edges, include_lowest=True)
        .astype("category")
        .cat.categories
    )


def _grid_fixed_edges(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
):
    d = df[[x, y, z]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d[z] = pd.to_numeric(d[z], errors="coerce")
    d = d.dropna()

    if d.empty:
        return None

    x_centers, x_ranges = _centers_and_ranges(x_edges)
    y_centers, y_ranges = _centers_and_ranges(y_edges)

    d["xbin"] = pd.cut(d[x], bins=x_edges, include_lowest=True)
    d["ybin"] = pd.cut(d[y], bins=y_edges, include_lowest=True)

    full_x = _full_categories_from_edges(x_edges)
    full_y = _full_categories_from_edges(y_edges)

    grid = d.pivot_table(index="ybin", columns="xbin", values=z, aggfunc="median", observed=True)
    cnt = d.pivot_table(index="ybin", columns="xbin", values=z, aggfunc="size", observed=True)

    grid = grid.reindex(index=full_y, columns=full_x)
    cnt = cnt.reindex(index=full_y, columns=full_x)

    Z = grid.values.astype(float)
    N = cnt.values.astype(float)

    # Flip y so larger y bins appear higher in the plot
    Z = Z[::-1, :]
    N = N[::-1, :]
    y_centers = y_centers[::-1]
    y_ranges = y_ranges[::-1]

    return Z, N, x_centers, y_centers, x_ranges, y_ranges


# -------------------------
# Recipe map heatmaps (3x2)
# -------------------------

def plot_recipe_maps_3x2(
    small_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    pairs=(
        ("cons8_median", "log_duration", "Consistency x Duration"),
        ("pace8_median", "log_duration", "Pace x Duration"),
        ("pace8_median", "cons8_median", "Pace x Consistency"),
    ),
    z_col: str = "residual",
    bins: int = 8,
    min_cell_n: int = 8,
    low_opacity: float = 0.30,
    q_low: float = 0.02,
    q_high: float = 0.98,
    scale_q: float | None = 0.98,
    viridis_wash: float = 0.35,
    z_widen: float = 1.0,
):
    """
    3x2 heatmaps (rows = feature pairs, cols = tiers) with faded low-N cells.

    - Global color scale is computed from reliable cells only (N >= min_cell_n).
    - Optional clipping by quantile scale_q on abs(z).
    - Washed Viridis used to reduce vividness.
    """
    tiers = [("small", small_df.copy()), ("mid", mid_df.copy())]

    for tier_name, d in tiers:
        if z_col not in d.columns:
            raise KeyError(f"{tier_name}_df missing required column: {z_col}")

    for x, y, _ in pairs:
        for tier_name, d in tiers:
            if x not in d.columns or y not in d.columns:
                raise KeyError(f"{tier_name}_df missing feature columns: {x}, {y}")

    all_df = pd.concat([small_df, mid_df], ignore_index=True)
    feats = sorted(set([x for x, _, _ in pairs] + [y for _, y, _ in pairs]))
    edges = {f: _bin_edges_global(all_df[f], bins=bins, q_low=q_low, q_high=q_high) for f in feats}

    grids: dict[tuple[str, str], tuple] = {}
    reliable_vals = []

    for tier_name, d in tiers:
        for x, y, title in pairs:
            out = _grid_fixed_edges(d, x, y, z_col, edges[x], edges[y])
            grids[(tier_name, title)] = out
            if out is None:
                continue

            Z, N, *_rest = out
            if min_cell_n and min_cell_n > 1:
                Zr = np.where(N >= min_cell_n, Z, np.nan)
            else:
                Zr = Z

            v = Zr[np.isfinite(Zr)]
            if v.size:
                reliable_vals.append(v)

    if not reliable_vals:
        zmin, zmax = -0.1, 0.1
    else:
        vv = np.concatenate(reliable_vals)
        a = np.abs(vv)
        m = float(np.nanmax(a)) if scale_q is None else float(np.nanquantile(a, scale_q))
        if not np.isfinite(m) or m <= 0:
            m = 0.1
        m *= float(z_widen) if z_widen and z_widen > 0 else 1.0
        zmin, zmax = -m, m

    colorscale = softened_viridis(viridis_wash)

    fig = make_subplots(
        rows=len(pairs),
        cols=2,
        column_titles=["Small tier", "Mid tier"],
        row_titles=[p[2] for p in pairs],
        horizontal_spacing=0.07,
        vertical_spacing=0.10,
    )

    colorbar_shown = False

    for r, (x, y, pair_title) in enumerate(pairs, start=1):
        for c, (tier_name, _d) in enumerate(tiers, start=1):
            out = grids.get((tier_name, pair_title))
            x_label = pretty_name(x)
            y_label = pretty_name(y)

            if out is None:
                fig.add_trace(go.Heatmap(z=[[None]], showscale=False, hoverinfo="skip"), row=r, col=c)
                fig.update_xaxes(title_text=x_label, row=r, col=c)
                fig.update_yaxes(title_text=y_label, row=r, col=c)
                continue

            Z, N, x_centers, y_centers, x_ranges, y_ranges = out

            if min_cell_n and min_cell_n > 1:
                Z_hi = np.where((N >= min_cell_n) & np.isfinite(Z), Z, np.nan)
                Z_lo = np.where((N > 0) & (N < min_cell_n) & np.isfinite(Z), Z, np.nan)
            else:
                Z_hi = Z.copy()
                Z_lo = np.full_like(Z, np.nan)

            hover = []
            for yi in range(len(y_centers)):
                ylo, yhi = y_ranges[yi]
                row_txt = []
                for xi in range(len(x_centers)):
                    xlo, xhi = x_ranges[xi]
                    zval = Z[yi, xi]
                    nval = N[yi, xi]
                    n_int = int(nval) if np.isfinite(nval) else 0

                    if (not np.isfinite(zval)) or n_int == 0:
                        row_txt.append(
                            f"<b>{tier_name}</b><br>"
                            f"{pair_title}<br>"
                            f"{x_label}: [{xlo:.2f}, {xhi:.2f}]<br>"
                            f"{y_label}: [{ylo:.2f}, {yhi:.2f}]<br>"
                            f"N: {n_int}<br>"
                            f"median {z_col}: -"
                        )
                    else:
                        tag = " (low N, faded)" if (min_cell_n and n_int < min_cell_n) else ""
                        row_txt.append(
                            f"<b>{tier_name}</b><br>"
                            f"{pair_title}{tag}<br>"
                            f"{x_label}: [{xlo:.2f}, {xhi:.2f}]<br>"
                            f"{y_label}: [{ylo:.2f}, {yhi:.2f}]<br>"
                            f"N: {n_int}<br>"
                            f"median {z_col}: {zval:.3f}<br>"
                            f"approx {(10**zval - 1)*100:.1f}% vs size expectation"
                        )
                hover.append(row_txt)

            want_scale = (not colorbar_shown) and (c == 2) and (r == 2)
            if want_scale:
                colorbar_shown = True

            if np.isfinite(Z_hi).any():
                fig.add_trace(
                    go.Heatmap(
                        z=Z_hi,
                        x=x_centers,
                        y=y_centers,
                        colorscale=colorscale,
                        zmin=zmin,
                        zmax=zmax,
                        hoverinfo="text",
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        opacity=1.0,
                        showscale=want_scale,
                        colorbar=dict(
                            title=f"{z_col}<br><span style='font-size:11px'>0 = as expected</span>",
                            x=1.03,
                            y=0.5,
                            len=0.88,
                        ) if want_scale else None,
                    ),
                    row=r,
                    col=c,
                )

            if np.isfinite(Z_lo).any():
                fig.add_trace(
                    go.Heatmap(
                        z=Z_lo,
                        x=x_centers,
                        y=y_centers,
                        colorscale=colorscale,
                        zmin=zmin,
                        zmax=zmax,
                        hoverinfo="text",
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        showscale=False,
                        opacity=low_opacity,
                    ),
                    row=r,
                    col=c,
                )

            fig.update_xaxes(title_text=x_label, tickformat=".2f", tickangle=45, row=r, col=c)
            fig.update_yaxes(title_text=y_label, tickformat=".2f", row=r, col=c)

    fig.update_layout(
        title=dict(text="Heatmaps of efficiency residuals by structural factor pairs", x=0.5),
        width=1200,
        height=900,
        margin=dict(l=140, r=190, t=90, b=60),
        **PLOTLY_THEME,
    )
    return fig


# -------------------------
# Effects bars
# -------------------------

def plot_effects_bars(
    effects_df: pd.DataFrame,
    tier_order=("small", "mid"),
    var_order=("log_duration", "cons8_median", "pace8_median"),
    eff_col="%ΔE_p25->p75",
    lo_col="%ΔE_CI_low",
    hi_col="%ΔE_CI_high",
    palette=BAR_PALETTE,
    show: bool = False,
):
    df = effects_df.copy()
    if df.empty:
        print("[warn] effects_df is empty")
        return None

    df["tier"] = df["tier"].astype(str)
    tiers = [t for t in tier_order if t in set(df["tier"])] or sorted(df["tier"].unique())

    df["var"] = df["var"].astype(str)
    vars_keep = [v for v in var_order if v in set(df["var"])] or sorted(df["var"].unique())

    df = df[df["tier"].isin(tiers) & df["var"].isin(vars_keep)].copy()
    y_map = {v: pretty_name(v) for v in vars_keep}

    fig = go.Figure()

    for i, tier in enumerate(tiers):
        dt = df[df["tier"] == tier].set_index("var").reindex(vars_keep)

        x = dt[eff_col].to_numpy(dtype=float)
        lo = dt[lo_col].to_numpy(dtype=float)
        hi = dt[hi_col].to_numpy(dtype=float)
        custom = np.column_stack([lo, hi])

        fig.add_trace(
            go.Bar(
                name=tier,
                y=[y_map[v] for v in vars_keep],
                x=x,
                orientation="h",
                marker=dict(color=palette[i % len(palette)]),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(hi - x),
                    arrayminus=(x - lo),
                    color="rgba(120,120,120,0.9)",
                    thickness=2,
                    width=6,
                ),
                customdata=custom,
                hovertemplate=(
                    "tier=%{fullData.name}<br>"
                    "var=%{y}<br>"
                    "effect=%{x:.3f}%<br>"
                    "CI=[%{customdata[0]:.3f}%, %{customdata[1]:.3f}%]<br>"
                    "<extra></extra>"
                ),
            )
        )

    all_lo = df[lo_col].to_numpy(dtype=float)
    all_hi = df[hi_col].to_numpy(dtype=float)
    xmin = float(np.nanmin(all_lo))
    xmax = float(np.nanmax(all_hi))
    pad = 0.10 * (xmax - xmin if xmax > xmin else 1.0)

    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        barmode="group",
        title="Structural factor effects by tier (p25 to p75, back-translated)",
        xaxis_title="% change in efficiency",
        yaxis_title="",
        xaxis=dict(range=[xmin - pad, xmax + pad]),
        height=420,
        width=950,
        margin=dict(l=110, r=40, t=60, b=40),
        legend_title_text="tier",
        **PLOTLY_THEME,
    )

    if show:
        fig.show()
    return fig


# -------------------------
# Coef bars with stars
# -------------------------

def _pval_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_coef_bars(
    coefs_df: pd.DataFrame,
    tier_order=("small", "mid"),
    var_order=("log_duration", "cons8_median", "pace8_median"),
    palette=BAR_PALETTE,
    show: bool = False,
):
    """
    Grouped horizontal bars with:
    - 95% CI whiskers using HC3 SE
    - significance stars placed after the right whisker (ci_high) for every tier
    """
    df = coefs_df.copy()
    if df.empty:
        print("[warn] coefs_df is empty")
        return None

    df["tier"] = df["tier"].astype(str)
    tiers = [t for t in tier_order if t in set(df["tier"])] or sorted(df["tier"].unique())

    df["var"] = df["var"].astype(str)
    vars_keep = [v for v in var_order if v in set(df["var"])] or sorted(df["var"].unique())

    df = df[df["tier"].isin(tiers) & df["var"].isin(vars_keep)].copy()

    df["ci_low"] = df["beta"] - 1.96 * df["se"]
    df["ci_high"] = df["beta"] + 1.96 * df["se"]
    df["stars"] = df["pval"].apply(_pval_stars)

    y_labels = [pretty_name(v) for v in vars_keep]

    xmin = float(np.nanmin(df["ci_low"].to_numpy(dtype=float)))
    xmax = float(np.nanmax(df["ci_high"].to_numpy(dtype=float)))
    xrng = (xmax - xmin) if xmax > xmin else 1.0

    xpad = 0.30 * xrng
    dx = 0.06 * xrng

    if len(tiers) == 1:
        tier_yshift = {tiers[0]: -22}
    elif len(tiers) == 2:
        tier_yshift = {tiers[0]: -22, tiers[1]: 22}
    else:
        shifts = np.linspace(-30, -18, num=len(tiers))
        tier_yshift = dict(zip(tiers, shifts))

    fig = go.Figure()

    for i, tier in enumerate(tiers):
        dt = df[df["tier"] == tier].set_index("var").reindex(vars_keep)

        beta = dt["beta"].to_numpy(dtype=float)
        lo = dt["ci_low"].to_numpy(dtype=float)
        hi = dt["ci_high"].to_numpy(dtype=float)
        p = dt["pval"].to_numpy(dtype=float)

        custom = np.column_stack([lo, hi, p])

        fig.add_trace(
            go.Bar(
                name=tier,
                y=y_labels,
                x=beta,
                orientation="h",
                marker=dict(color=palette[i % len(palette)]),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(hi - beta),
                    arrayminus=(beta - lo),
                    color="rgba(120,120,120,0.9)",
                    thickness=2,
                    width=6,
                ),
                customdata=custom,
                hovertemplate=(
                    "tier=%{fullData.name}<br>"
                    "var=%{y}<br>"
                    "beta=%{x:.6f}<br>"
                    "CI=[%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>"
                    "p=%{customdata[2]:.3g}"
                    "<extra></extra>"
                ),
            )
        )

    for tier in tiers:
        dt = df[df["tier"] == tier].set_index("var").reindex(vars_keep)
        for v, ylab in zip(vars_keep, y_labels):
            star = dt.loc[v, "stars"]
            if not isinstance(star, str) or star == "":
                continue

            x_star = float(dt.loc[v, "ci_high"]) + dx
            fig.add_annotation(
                x=x_star,
                y=ylab,
                xref="x",
                yref="y",
                text=star,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                yshift=float(tier_yshift[tier]),
                font=dict(size=14),
            )

    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        barmode="group",
        title="WLS coefficients by tier (HC3)  (* p<0.05, ** p<0.01, *** p<0.001)",
        xaxis_title="coefficient (HC3 95% CI)",
        yaxis_title="",
        xaxis=dict(range=[xmin - xpad, xmax + xpad]),
        height=420,
        width=950,
        margin=dict(l=110, r=40, t=60, b=40),
        legend_title_text="tier",
        **PLOTLY_THEME,
    )

    if show:
        fig.show()
    return fig


# -------------------------
# Example usage
# -------------------------
# fig = plot_corr_small_mid(small_df, mid_df, pace_col="pace8_median", cons_col="cons8_median", show=True)
# fig.write_html("corr.html")
#
# fig2 = plot_recipe_maps_3x2(small_df, mid_df, viridis_wash=0.45, z_widen=1.3)
# fig2.write_html("maps.html")
#
# fig3 = plot_effects_bars(effects_df, show=True)
# if fig3 is not None:
#     fig3.write_html("effects.html")
#
# fig4 = plot_coef_bars(coefs_df, show=True)
# if fig4 is not None:
#     fig4.write_html("coefs.html")
