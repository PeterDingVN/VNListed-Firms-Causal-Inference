import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from scipy.stats import norm


def cluster_lasso(X, y, c=1.1, max_iter=100, tol=1e-6):
    """
    Cluster Lasso for feature selection in HD-Panel data. The construction is based on Belloni (2015).

    Parameters:
    - X: features matrix. Must be DataFrame or Series
    - y: target. Must share index with X. Could be dataframe or series
    - c: Constant > 1 (default based on article)
    - gamma: Significance level between 0 and 1 (recommended 0.1/log(max(no_features; no_observation))).
    - max-iter: number of iteration for updating Cluster Lasso estimator Beta

    Returns:
    - selected_vars: List of variables selected by the model.
    """



    X = X.copy().set_index(['company', 'year']).sort_index()
    y = y.copy().set_index(['company', 'year']).sort_index()

    # Key params
    N_obs = len(X)
    p = X.shape[1]  # Number of features

    # =========================================================================
    # STEP 1: Demean X and y by individual within group mean
    # =========================================================================

    # y_dot_it = y_it - mean(y_i)
    y_mean = y.groupby(level=0).transform('mean')
    y_dot = y - y_mean

    # X_dot_it = X_it - mean(X_i)
    X_mean = X.groupby(level=0).transform('mean')
    X_dot = X - X_mean


    # =========================================================================
    # STEP 2: Iterative Cluster Lasso
    # =========================================================================

    # Initial Phi and lambda

    # Formula: lambda = 2 * c * sqrt(nT) * invPhi(1 - gamma/2p)
    # Formula: phi_j = Phi_j = sqrt( 1/nT * sum_i ( sum_t (x_dot_itj * eps_it) )^2

    residuals = y_dot.copy()
    gamma = 0.1/np.log(max(p, N_obs))
    Phi_inv = norm.ppf(1 - gamma / (2 * p))

    # Lambda
    lambda_param = 2 * c * np.sqrt(N_obs) * Phi_inv

    selected_indices = []

    for k in range(max_iter):
        # Phi_j

        # 1. Compute product of X and residuals element-wise
        product_term = X_dot.multiply(residuals.squeeze(), axis=0)

        # 2. Sum over time T (groupby Entity/Group)
        sum_over_t = product_term.groupby(level=0).sum()

        # 3. Square the sums, sum over groups i, divide by total obs, take sqrt
        phi_loadings = np.sqrt((sum_over_t ** 2).sum() / N_obs)

        # Handle cases where phi might be zero (drop those or set small epsilon)
        phi_loadings = phi_loadings.replace(0, 1e-8)

    # C. Run Weighted Lasso
        # Scale X by penalty loadings
        X_scaled = X_dot / phi_loadings
        alpha_val = lambda_param / N_obs

        # Run Lasso
        # fit_intercept=False because data is already demeaned
        lasso_model = Lasso(alpha=alpha_val, fit_intercept=False, tol=tol, selection='random')
        lasso_model.fit(X_scaled, y_dot)

        # Transform coefficients back: Beta = Beta_new / Phi
        beta_current = pd.Series(lasso_model.coef_, index=X.columns) / phi_loadings

        # D. Update Residuals
        # Formula: e_it = y_dot_it - x_dot_it * Beta
        residuals = y_dot.squeeze() - X_dot.dot(beta_current)

        # E. Check Stability / Convergence (Condition R')
        # -----------------------------------------------
        # Check if the set of selected variables (support) has stabilized
        new_selected_indices = beta_current[beta_current > 1e-8].index.tolist()

        if set(new_selected_indices) == set(selected_indices) and k > 0:
            print(f"Converged at iteration {k + 1}")
            selected_indices = new_selected_indices
            break

        selected_indices = new_selected_indices

    else:
        print("Max iterations reached.")

    # =========================================================================
    # STEP 3: Post-Lasso to debias coef of selected features
    # =========================================================================

    # Select only the features that survived Lasso (coef != 0)
    final_features = selected_indices

    if len(final_features) == 0:
        print("No features selected. Returning zero coefficients.")

    # Slice the demeaned data to only selected features
    X_subset = X_dot[final_features]

    # Run OLS (Ordinary Least Squares) on selected subset
    # This removes the shrinkage bias introduced by the Lasso penalty
    post_lasso_model = LinearRegression(fit_intercept=False)
    post_lasso_model.fit(X_subset, y_dot)

    # Construct final result series
    beta_final = pd.Series(0.0, index=X.columns)
    beta_final[final_features] = post_lasso_model.coef_

    return final_features