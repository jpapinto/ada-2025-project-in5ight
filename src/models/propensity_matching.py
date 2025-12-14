import numpy as np
import pandas as pd
from statsmodels.formula.api import logit
from sklearn.neighbors import NearestNeighbors

def estimate_propensity_scores(df, formula, treatment_col='is_outperformer'):
    """
    Fits a logistic regression to estimate propensity scores.
    """
    # Ensure treatment is integer
    df_fit = df.copy()
    df_fit[treatment_col] = df_fit[treatment_col].astype(int)
    
    print(f"\nFitting Propensity Score Model: {formula}")
    model = logit(formula=formula, data=df_fit).fit_regularized(
        L1_wt=0.0, alpha=0.1, disp=False, maxiter=1000
    )
    
    return model.predict(df_fit)

def match_nearest_neighbor_within_category(treated_df, control_df, exact_match_col, propensity_col='propensity_score', caliper=0.05):
    """
    Performs 1:1 Nearest Neighbor matching with replacement within categories.
    """
    print(f"Starting matching (Treated={len(treated_df)}, Control={len(control_df)})...")
    
    matched_pairs = []
    used_control_indices = set()
    
    categories = treated_df[exact_match_col].unique()
    
    for cat in categories:
        # Subset by category (Exact Match)
        t_cat = treated_df[treated_df[exact_match_col] == cat]
        c_cat = control_df[control_df[exact_match_col] == cat]
        
        if c_cat.empty:
            continue
            
        # KNN on Propensity Score
        X_control = c_cat[[propensity_col]].values
        X_treated = t_cat[[propensity_col]].values
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_control)
        distances, indices = nbrs.kneighbors(X_treated)
        
        # Greedy assignment
        candidates = []
        for i in range(len(t_cat)):
            dist = distances[i][0]
            c_iloc = indices[i][0]
            if dist <= caliper:
                candidates.append((dist, t_cat.index[i], c_cat.index[c_iloc]))
        
        candidates.sort(key=lambda x: x[0])
        
        for dist, t_idx, c_idx in candidates:
            if c_idx not in used_control_indices:
                matched_pairs.append(t_idx)
                matched_pairs.append(c_idx)
                used_control_indices.add(c_idx)

    return set(matched_pairs)