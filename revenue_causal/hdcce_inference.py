import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path
from scipy.stats import norm


def hdcce_inference(data, id_col, time_col, y_col, x_cols,
                    target_col_index, # Used for nodewise regression
                    trunc=0.01,
                    n_factors=None,
                    n_folds=10,
                    alpha_sig = None,
                    hac_method=2,
                    random_state=None):
    """
    Python implementation of HD-CCE Inference (Linton et al., 2024).
    Adapted for UNBALANCED panels.

    Logic aligns with original R implementation:
    Step 1: Estimate Factors (K) via PCA on Cross-Sectional Averages.
    Step 2: Project data (defactor) and run Main Lasso & Nodewise Lasso.
    Step 3: Variance-based penalty selection and Desparsification.
    Step 4: Inference (Confidence Bands).
    """

    # --- 0. Initial Checks & Data Prep ---
    if alpha_sig is None:
        alpha_sig = [0.01, 0.05, 0.1]
    df = data.copy()

    # Sort for consistent indexing (crucial for time-series structure)
    df = df.sort_values(by=[id_col, time_col])

    # Drop rows with missing values (Standard CCE requirement)
    # Note: In unbalanced panels, we drop rows with NaNs, but we keep the structure
    # where different units have different time lengths.
    cols_to_check = [y_col] + x_cols
    df = df.dropna(subset=cols_to_check)

    # Extract structural info
    ids = df[id_col].unique()
    valid_times = np.sort(df[time_col].unique())  # Global time set

    target_col = x_cols[target_col_index]

    # ============================================================================#
    #  Estimation of number of factors K_hat (Step 1 in R)
    # ============================================================================#

    # R Code: X_bar via colMeans loop.
    # Python Unbalanced: Groupby mean to handle missing units at specific times.
    X_bar_df = df.groupby(time_col)[x_cols].mean()

    # Align X_bar to valid times (T_valid x p)
    X_bar_df = X_bar_df.reindex(valid_times).dropna()
    valid_times_filtered = X_bar_df.index.values
    T_factor = len(valid_times_filtered)

    if T_factor < 2:
        raise ValueError("Insufficient time periods to estimate factors.")

    # Get X_bar EXCLUDING the target regressor (matching R logic: X_bar[,-COEF_INDEX])
    X_bar_no_j = X_bar_df.drop(columns=target_col).values

    # Empirical covariance matrix
    # R Code: Cov_X_bar <- 1/obs_T * t(X_bar) %*% X_bar
    Cov_X_bar = (1 / T_factor) * (X_bar_no_j.T @ X_bar_no_j)

    # Eigen decomposition
    # R Code: eigen(Cov_X_bar, symmetric = TRUE)
    eig_vals, eig_vecs = np.linalg.eigh(Cov_X_bar)

    # Sort descending (eigh returns ascending)
    idx_sorted = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx_sorted]
    eig_vecs = eig_vecs[:, idx_sorted]

    # Normalize
    if eig_vals[0] > 0:
        eig_vals_norm = eig_vals / eig_vals[0]
    else:
        eig_vals_norm = np.zeros_like(eig_vals)

    # Determine K_hat
    if n_factors is not None:
        K_hat = n_factors
    else:
        K_hat = np.sum(eig_vals_norm > trunc)
        K_hat = max(int(K_hat), 1)  # Safety: assume at least 1 factor usually

    # ============================================================================#
    # Step 1 c): Computation of projection matrix (Loadings)
    # ============================================================================#

    # R Code: W_tilde <- X_bar[,-COEF_INDEX] %*% vectors[,1:K_hat]
    W_tilde_global = X_bar_no_j @ eig_vecs[:, :K_hat]

    # Map time values to indices for easy lookup
    time_to_idx = {t: i for i, t in enumerate(valid_times_filtered)}

    # ============================================================================#
    # Step 2 a): Transform the data (Projection)
    # ============================================================================#
    # Note: R code computes one global Pi_tilde.
    # For UNBALANCED data, Pi must be computed per unit based on available times.

    y_tilde_list = []
    x_tilde_list = []

    # We store metadata to handle HAC grouping later
    # (Equivalent to R's implicit indexing via loop 1:Obs_N)
    meta_ids = []

    for unit_id in ids:
        # Slice data for this unit
        unit_mask = df[id_col] == unit_id
        unit_times = df.loc[unit_mask, time_col].values

        # Find which global time indices this unit possesses
        # (Intersection of unit's times and the times used for factors)
        global_indices = [time_to_idx[t] for t in unit_times if t in time_to_idx]

        # Check sufficient data: Need T_i > K_hat to project
        if len(global_indices) <= K_hat + 1:
            continue

        # Construct Unit-Specific W_i and Pi_i
        W_i = W_tilde_global[global_indices, :]  # (T_i, K)

        # Pi_i = I - W_i (W_i' W_i)^-1 W_i'
        # Using pseudo-inverse for stability
        try:
            WtW_inv = np.linalg.pinv(W_i.T @ W_i)
            P_i = W_i @ WtW_inv @ W_i.T
            M_i = np.eye(len(global_indices)) - P_i  # The projection matrix
        except np.linalg.LinAlgError:
            continue

        # Project Data
        y_unit = df.loc[unit_mask, y_col].values
        X_unit = df.loc[unit_mask, x_cols].values

        # Apply Projection
        y_tilde_list.append(M_i @ y_unit)
        x_tilde_list.append(M_i @ X_unit)
        meta_ids.extend([unit_id] * len(y_unit))

    if not y_tilde_list:
        raise ValueError("No units had sufficient data length > K_hat.")

    Y_tilde = np.concatenate(y_tilde_list)
    X_tilde = np.concatenate(x_tilde_list)
    unit_ids_arr = np.array(meta_ids)
    N_obs_total = len(Y_tilde)

    # ============================================================================#
    # Step 2 b): Main Lasso Regression
    # ============================================================================#
    # R uses cv.glmnet. Python uses LassoCV.
    # We use random K-Fold (cv=n_folds) which is standard for large N.

    main_lasso = LassoCV(cv=n_folds, random_state=random_state, fit_intercept=False)
    main_lasso.fit(X_tilde, Y_tilde)

    yhat_Lasso = main_lasso.predict(X_tilde)
    resid_Lasso = Y_tilde - yhat_Lasso

    # ============================================================================#
    # Step 2 b) Part 2: Nodewise Lasso
    # ============================================================================#

    X_target = X_tilde[:, target_col_index]  # x_j
    X_others = np.delete(X_tilde, target_col_index, axis=1)  # x_-j

    # Calculate full path to replicate R's variance loop
    alphas_node, coefs_node_path, _ = lasso_path(X_others, X_target, cv=n_folds, fit_intercept=False)

    # Find the alpha corresponding to CV min (lambda.min)
    # (We run a quick CV fit to get the best alpha, then map it to the path indices)
    node_cv_fit = LassoCV(cv=n_folds, fit_intercept=False)
    node_cv_fit.fit(X_others, X_target)
    kappa_cv_idx = np.argmin(np.abs(alphas_node - node_cv_fit.alpha_))

    # ============================================================================#
    # Step 3 a): Choose nodewise penalty parameter (Variance Scaling)
    # ============================================================================#

    kappa_grid_len = len(alphas_node)
    var_scaled = np.zeros(kappa_grid_len)

    # Pre-calculate Sigma estimates to speed up the loop (Vectorization)
    # This replaces the inner loop over i=1:obs_N in the R code

    # Calculate RSS per unit for HAC correction
    # Create a helper dataframe for grouping
    df_resid = pd.DataFrame({'unit': unit_ids_arr, 'resid_sq': resid_Lasso ** 2})
    unit_stats = df_resid.groupby('unit').agg(rss=('resid_sq', 'sum'), T=('resid_sq', 'count'))

    # Calculate Sigma_eps_estimate per unit (Step 3a HAC=2 logic)
    # R Formula: (1/T) * (T / (T-K)) * sum(resid^2)
    unit_stats['dof_adj'] = unit_stats['T'] / (unit_stats['T'] - K_hat).clip(lower=1)  # Safety clip
    unit_stats['sigma_sq'] = (1 / unit_stats['T']) * unit_stats['dof_adj'] * unit_stats['rss']

    # Map back to observation level for vectorized multiplication
    sigma_sq_vec = pd.Series(unit_ids_arr).map(unit_stats['sigma_sq']).values

    # --- Loop over alphas (kappa grid) ---
    for k in range(kappa_grid_len):
        coef_k = coefs_node_path[:, k]
        resid_node_Lasso = X_target - (X_others @ coef_k)

        # Denominator: (X_j' * resid_node)^2
        denom = (X_target.T @ resid_node_Lasso) ** 2
        if denom < 1e-12:
            var_scaled[k] = np.inf
            continue

        # Numerator calculation based on HAC method
        num = 0

        if hac_method == 1:  # Homoscedastic
            # R: sigma_eps <- (1/NT) * (T/T-K) * sum(resid^2) -> Global estimate
            # Approx global DoF scalar
            dof_global = N_obs_total / (N_obs_total - K_hat)
            sigma_global = np.mean(resid_Lasso ** 2) * dof_global
            num = sigma_global * np.sum(resid_node_Lasso ** 2)

        elif hac_method == 2:  # Heteroscedastic (Default)
            # R: sum_i { sigma_i * sum(resid_node_i^2) }
            # Vectorized: sum(sigma_sq_vec * resid_node^2)
            num = np.sum(sigma_sq_vec * (resid_node_Lasso ** 2))

        elif hac_method == 3:  # HAC
            # R: sum_i { (resid_node_i * resid_main_i)' W (resid_node_i * resid_main_i) }
            # Where W is matrix of 1s. This implies [sum(resid_node * resid_main)]^2

            # Element-wise product
            prod_term = resid_node_Lasso * resid_Lasso
            # Sum by unit, then square the sums
            group_sums = pd.Series(prod_term).groupby(unit_ids_arr).sum().values
            num = np.sum(group_sums ** 2)

        var_scaled[k] = num / denom

    # --- Thresholding Logic (Step 3a end) ---
    # R Code: V_TRUNC = 1.25 * var_scaled[kappa_cv_idx]
    V_TRUNC = 1.25 * var_scaled[kappa_cv_idx]

    # R Loop: Breaks when var_scaled > V_TRUNC.
    # Note: R glmnet returns lambdas high->low. Python lasso_path usually high->low.
    # We follow the R loop logic exactly.
    kappa_idx = 0
    for l in range(kappa_grid_len):
        if var_scaled[l] <= V_TRUNC:
            kappa_idx = l
        # The R code breaks if it exceeds.
        # However, checking the R code provided:
        # if(var_scaled[l] > V_TRUNC) { break }
        # This implies we take the last index that satisfied it (if sorted)
        # or we stop searching once we violate.
        if var_scaled[l] > V_TRUNC:
            break

    # ============================================================================#
    # Step 3 b): Construction of Desparsified Estimator
    # ============================================================================#

    # Get residuals for selected alpha
    coef_final = coefs_node_path[:, kappa_idx]
    resid_node_final = X_target - (X_others @ coef_final)

    # Formula: b_j = beta_lasso + (resid_node' resid_main) / (resid_node' X_j)
    beta_Lasso_j = main_lasso.coef_[target_col_index]

    numerator = resid_node_final.T @ resid_Lasso
    denominator = resid_node_final.T @ X_target

    despar_beta = beta_Lasso_j + (numerator / denominator)

    # ============================================================================#
    # Step 4: Inference
    # ============================================================================#

    Avar = np.sqrt(var_scaled[kappa_idx])

    ci_bands = []
    for a in alpha_sig:
        z = norm.ppf(1 - a / 2)
        ci_bands.append({
            'alpha': a,
            'lower': despar_beta - z * Avar,
            'upper': despar_beta + z * Avar
        })

    return {
        'coef_despar': despar_beta,
        'avar': Avar,
        'confidence_band': pd.DataFrame(ci_bands),
        'factors_estimated': K_hat,
        'selected_alpha_idx': kappa_idx
    }

