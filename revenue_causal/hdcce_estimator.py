import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


# Enable metadata routing globally
sklearn.set_config(enable_metadata_routing=True)

def hdcce_estimator(df, unit_col, time_col, y_col, x_cols,
                    TRUNC=0.01, NFACTORS=None, variant="Lasso",
                    lambda_val=None, NFOLDS=10,
                    foldid=None, scree_plot=True,
                    standardize=True):
    """
    Estimates High-Dimensional Panel Data Models with Interactive Fixed Effects (HD-CCE).
    Adapted to handle Unbalanced Panel Data.

    Parameters:
    -----------
    df : pd.DataFrame
        Long-format DataFrame containing the panel data.
    unit_col : str
        Name of the column identifying cross-section units (i).
    time_col : str
        Name of the column identifying time periods (t).
    y_col : str
        Name of the dependent variable column.
    x_cols : list of str
        List of regressor column names.
    TRUNC : float, optional
        Truncation parameter for estimating number of factors. Default 0.01.
    NFACTORS : int, optional
        Manual number of factors. If None, estimated via TRUNC.
    variant : str, optional
        "Lasso" (default) or "LS".
    lambda_val : float, optional
        User specified lambda (alpha in sklearn). If None, CV is used.
    NFOLDS : int, optional
        Number of folds for Cross-Validation.
        Note: Folds are created based on Units to preserve panel structure.
    foldid : array-like, optional
        User-defined fold labels.
    scree_plot : bool, optional
        Whether to display the scree plot of eigenvalues.
    standardize : bool, optional
        Whether to standardize regressors before Lasso. Default True.

    Returns:
    --------
    dict containing:
        - 'coefs': Estimated coefficients (on original scale)
        - 'K_hat': Estimated/Used number of factors
        - 'Lambda': Selected lambda (if Lasso variant)
    """

    # -------------------------------------------------------------------------
    # 0. Initial Data Preparation & Checks
    # -------------------------------------------------------------------------
    if variant not in ["LS", "Lasso"]:
        raise ValueError('Variant must be set to "LS" or "Lasso".')

    # Sort data to ensure alignment logic holds where necessary
    data = df.copy().sort_values(by=[unit_col, time_col])

    # Extract unique time periods (T) and regressors (P)
    unique_times = data[time_col].unique()
    unique_times.sort()  # Ensure time is ordered
    obs_T_global = len(unique_times)
    p = len(x_cols)

    # Check dimensions
    if NFACTORS is not None and int(NFACTORS) >= p:
        raise ValueError("NFACTORS must be less than the number of regressors (p).")

    # -------------------------------------------------------------------------
    # Step 1: Eigendecomposition of Empirical Covariance Matrix (X_bar)
    # -------------------------------------------------------------------------
    # For unbalanced data, X_bar_t is the mean of available units at time t.
    # We group by time and take the mean of X columns.

    X_bar_df = data.groupby(time_col)[x_cols].mean()

    # Ensure X_bar is aligned with sorted unique_times (fill missing times if any gaps exist in index)
    X_bar_df = X_bar_df.reindex(unique_times)

    # If a time period has NO observations across all units, we cannot estimate factors for it.
    # We drop those times for factor estimation purposes.
    valid_times_mask = X_bar_df.notna().all(axis=1)
    if not valid_times_mask.all():
        print(f"Warning: {obs_T_global - valid_times_mask.sum()} time periods dropped due to zero observations.")
        X_bar_df = X_bar_df[valid_times_mask]

    X_bar_matrix = X_bar_df.values  # Shape (T_valid, P)
    T_valid = X_bar_matrix.shape[0]

    # Covariance Matrix: (1/T) * X_bar' @ X_bar
    Cov_X_bar = (1 / T_valid) * (X_bar_matrix.T @ X_bar_matrix)

    # Eigen Decomposition (eigh for symmetric)
    # Numpy returns eigenvalues smallest -> largest. We reverse to match R logic.
    eigen_vals, eigen_vecs = np.linalg.eigh(Cov_X_bar)
    eigen_vals = eigen_vals[::-1]
    eigen_vecs = eigen_vecs[:, ::-1]

    # -------------------------------------------------------------------------
    # Step 2: Estimation of Number of Factors
    # -------------------------------------------------------------------------
    # Normalize with the largest eigenvalue
    eigen_values_norm = eigen_vals / eigen_vals[0]

    if NFACTORS is not None:
        K_hat = int(NFACTORS)
        msg_title = f"User-supplied K = {K_hat}"
    else:
        # Count eigenvalues > TRUNC
        K_hat = np.sum(eigen_values_norm > TRUNC)
        msg_title = f"Estimated K = {K_hat}"

    print(f"{msg_title}")

    if scree_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(eigen_values_norm, 'o-', markersize=4, label='Eigenvalues')
        plt.axhline(y=TRUNC, color='r', linestyle='--', label=f'Truncation ({TRUNC})')
        plt.title(f"Scree Plot: {msg_title}")
        plt.ylabel("Normalized Eigenvalues")
        plt.xlabel("Factor Index")
        plt.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # Step 3: Computation of Global Factors (W_hat)
    # -------------------------------------------------------------------------
    # W_hat = X_bar * Gamma_hat (vectors corresponding to top K factors)
    # Shape: (T_valid, K_hat)
    W_hat_matrix = X_bar_matrix @ eigen_vecs[:, :K_hat]

    # Create a lookup series for factors indexed by time
    # This allows us to map specific times in unbalanced panel to the correct factor row
    W_hat_df = pd.DataFrame(W_hat_matrix, index=X_bar_df.index, columns=[f'F{k}' for k in range(K_hat)])

    # -------------------------------------------------------------------------
    # Step 4: Transform the Data (Projection) - Unbalanced Adaptation
    # -------------------------------------------------------------------------
    # Original R logic: Y_hat = Pi_hat * Y.
    # Pi_hat projects onto orthogonal complement of W.
    # Mathematically equivalent to taking residuals of Y regressed on W.
    # For unbalanced data, we must do this regression unit-by-unit using only
    # the time periods where that unit is active.

    Y_transformed_list = []
    X_transformed_list = []

    # To preserve groups for CV later
    groups_list = []

    # We iterate by unit to handle specific time masks
    for unit_id, group in data.groupby(unit_col):
        # Align unit data with global time to find intersection
        # Extract y and X for this unit
        y_unit = group[y_col].values
        X_unit = group[x_cols].values

        # Get the factors W corresponding to this unit's time periods
        unit_times = group[time_col]

        # Intersection of unit times and times where we successfully estimated factors
        valid_idx = unit_times.isin(W_hat_df.index)

        if not valid_idx.any():
            continue

        # Filter data to valid times
        y_unit = y_unit[valid_idx]
        X_unit = X_unit[valid_idx]
        current_times = unit_times[valid_idx]

        # Extract W for these specific times
        W_unit = W_hat_df.loc[current_times].values

        if K_hat > 0:
            # Projection: Residuals of regression on W
            # We use OLS to project out W.
            # Fit intercept=False because factors usually capture the mean or we centered X_bar.
            # (R code Pi_hat construction implies no intercept in projection matrix)

            # 1. Project Y
            reg_y = LinearRegression(fit_intercept=False)
            reg_y.fit(W_unit, y_unit)
            y_tilde = y_unit - reg_y.predict(W_unit)

            # 2. Project X (each column)
            reg_x = LinearRegression(fit_intercept=False)
            reg_x.fit(W_unit, X_unit)
            X_tilde = X_unit - reg_x.predict(W_unit)
        else:
            # If 0 factors, no transformation
            y_tilde = y_unit
            X_tilde = X_unit

        Y_transformed_list.append(y_tilde)
        X_transformed_list.append(X_tilde)

        # Record unit ID for every observation for grouping in CV
        groups_list.append(np.full(len(y_tilde), unit_id))

    # Stack all transformed data
    Y_final = np.concatenate(Y_transformed_list)
    X_final = np.vstack(X_transformed_list)
    groups_final = np.concatenate(groups_list)

    # -------------------------------------------------------------------------
    # Step 5: Estimate Lasso or OLS on Transformed Data
    # -------------------------------------------------------------------------

    results = {
        'coefs': None,
        'K_hat': K_hat,
        'Lambda': None
    }

    # -- Variant: Least Squares --
    if variant == "LS":
        # Simply run OLS on transformed data
        reg_ls = LinearRegression(fit_intercept=False)
        reg_ls.fit(X_final, Y_final)
        results['coefs'] = reg_ls.coef_
        print("LS variant estimated.")

    # -- Variant: Lasso --
    elif variant == "Lasso":

        # 1. Manual Standardization (Crucial for Lasso in Python)
        scaler = None
        X_train = X_final

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_final)

        # 2. Estimator Setup
        if lambda_val is not None:
            # User supplied lambda
            model = Lasso(alpha=lambda_val, fit_intercept=False, random_state=42)
            model.fit(X_train, Y_final)
            final_coefs_scaled = model.coef_
            results['Lambda'] = lambda_val
            print("User specified lambda used.")

        else:
            # Cross-Validation
            # We must respect panel structure (Cluster by Unit)
            cv_splitter = None

            if foldid is not None:
                # User provided folds (Must match length of stacked data)
                if len(foldid) != len(Y_final):
                    raise ValueError("Length of foldid does not match number of valid observations.")
                from sklearn.model_selection import PredefinedSplit
                cv_splitter = PredefinedSplit(test_fold=foldid)
                print("User specified foldid used.")
            else:
                # GroupKFold ensures no unit is in both train and test set
                cv_splitter = GroupKFold(n_splits=NFOLDS)
                print(f"GroupKFold (NFOLDS={NFOLDS}) used.")

            model = LassoCV(cv=cv_splitter, fit_intercept=False, random_state=42, n_jobs=-1)

            # LassoCV.fit requires groups arg if using GroupKFold
            model.fit(X_train, Y_final, groups=groups_final)

            final_coefs_scaled = model.coef_
            results['Lambda'] = model.alpha_
            print(f"CV complete. Lambda min: {model.alpha_:.6f}")

        # 3. Rescaling Coefficients (if standardized)
        if standardize and scaler is not None:
            # Beta_orig = Beta_scaled / sigma_x
            scale_factors = scaler.scale_
            # Safety for constant columns (scale=0)
            scale_factors[scale_factors == 0] = 1.0
            results['coefs'] = final_coefs_scaled / scale_factors
        else:
            results['coefs'] = final_coefs_scaled

    return results
