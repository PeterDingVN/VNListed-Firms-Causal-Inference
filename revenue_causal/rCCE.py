import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power, inv


def regularized_cce_unbalanced(df, y_col, x_cols, id_cols, r_max=None):
    """
    Implements Regularized CCE for Unbalanced Panel Data.

    Parameters:
    - df: Pandas DataFrame with MultiIndex (entity, time).
          Must contain columns for y_col and x_cols.
    - y_col: String name of dependent variable.
    - x_cols: List of string names of regressors.
    - r_max: Maximum number of factors to test (default: K_z - 1).

    Returns:
    - beta_pooled: The Regularized CCE Pooled Estimator.
    - beta_mg: The Regularized CCE Mean-Group Estimator.
    """

    # ---------------------------------------------------------
    # PREPARATION
    # Define Z_i = [y_i, X_i] (The proxies)
    # ---------------------------------------------------------

    z_cols = [y_col] + x_cols + id_cols
    Z = df[z_cols]

    # Ensure data is sorted for time-series operations
    Z = Z.set_index(['company', 'year'])
    Z = Z.sort_index()

    # K_z: Number of proxies
    K_z = len(z_cols)
    if r_max is None:
        r_max = K_z - 1

    # ---------------------------------------------------------
    # STEP 1: Construct Adapted Covariance Matrix (Sigma_hat)
    # ---------------------------------------------------------
    # 1.1 Calculate Unbalanced Cross-Sectional Averages (Z_bar_t)
    # Pandas .mean() automatically skips NaNs, handling the N_t variation.
    Z_bar = Z.groupby(level='year').mean()

    # 1.2 Calculate Residuals (Z_it - Z_bar_t)
    # Pandas automatically aligns the time index when subtracting.
    # We drop NaNs to ensure we only sum over available observations (S_t).
    residuals = (Z - Z_bar).dropna()

    # 1.3 Compute Covariance Matrix
    # Formula: sum(residuals' * residuals) / M_total
    # Note: We use .values to switch to numpy for matrix multiplication
    resid_mat = residuals.values
    M_total = len(residuals)  # Total number of observed unit-time pairs

    Sigma_hat = (resid_mat.T @ resid_mat) / M_total

    # ---------------------------------------------------------
    # STEP 2: Construct Normalized Factor Proxies (F_hat)
    # ---------------------------------------------------------
    # Formula: F_hat = Z_bar * (Sigma_hat^(-1/2))'

    # Compute Sigma^(-1/2)
    # We use fractional_matrix_power from scipy
    Sigma_inv_sqrt = fractional_matrix_power(Sigma_hat, -0.5)

    # Ensure it's real (numerical noise can cause negligible imaginary parts)
    Sigma_inv_sqrt = np.real(Sigma_inv_sqrt)

    # Apply normalization
    # Z_bar is T x K_z, Sigma_inv_sqrt is K_z x K_z
    F_hat = Z_bar.dot(Sigma_inv_sqrt.T)

    # ---------------------------------------------------------
    # STEP 3: Estimate Number of Factors (R_hat)
    # ---------------------------------------------------------
    # Using Eigenvalue Ratio (ER) criterion

    T = len(F_hat)
    F_mat = F_hat.values

    # Matrix: T^(-1) * F'F
    inner_mat = (F_mat.T @ F_mat) / T

    # Get eigenvalues (sorted descending)
    # eigh returns ascending, so we reverse
    eigvals = np.linalg.eigvalsh(inner_mat)[::-1]

    # Calculate ER(r) = v_r / v_{r+1}
    # We check r from 1 to r_max
    er_ratios = []
    possible_rs = range(1, r_max + 1)

    for r in possible_rs:
        # Indices in python are 0-based. v_r is index r-1.
        v_r = eigvals[r - 1]
        v_r_next = eigvals[r]

        if v_r_next < 1e-9:  # Avoid division by zero
            ratio = 0
        else:
            ratio = v_r / v_r_next
        er_ratios.append(ratio)

    # Select R that maximizes ratio
    # +1 because python index 0 corresponds to r=1
    R_hat = np.argmax(er_ratios) + 1

    print(f"Estimated number of factors (R_hat): {R_hat}")

    # ---------------------------------------------------------
    # STEP 4: Construct Regularized Factors (F_r)
    # ---------------------------------------------------------
    # We need the eigenvectors corresponding to the largest eigenvalues
    # of the T x T matrix or use SVD on F_hat directly.
    # The paper mentions: F_r = sqrt(T) * U_R

    # SVD of F_hat (T x K_z) -> U * S * V.T
    # We want the first R_hat columns of U
    U, S, Vt = np.linalg.svd(F_mat, full_matrices=False)

    # Extract first R_hat columns
    U_R = U[:, :R_hat]

    # Scale by sqrt(T)
    F_r_values = np.sqrt(T) * U_R

    # Put back into Pandas for index alignment later
    F_r = pd.DataFrame(F_r_values, index=F_hat.index)

    # ---------------------------------------------------------
    # STEP 5: Calculate Regularized Estimators (Unbalanced)
    # ---------------------------------------------------------

    # Accumulators for Pooled Estimator
    numerator_pooled = np.zeros((len(x_cols), 1))
    denominator_pooled = np.zeros((len(x_cols), len(x_cols)))

    # List for Mean Group estimates
    betas_mg = []

    # Iterate over every unit i
    entities = df.index.get_level_values(level='company').unique()

    for i in entities:
        # Get data for unit i
        # dropna() handles the unbalanced availability
        unit_data = df.loc[i].dropna(subset=[y_col] + x_cols)

        if unit_data.empty:
            continue

        y_i = unit_data[[y_col]].values
        X_i = unit_data[x_cols].values

        # Get the Regularized Factors for the SAME time periods as unit i
        # This is the "Projection Matrix constructed using only rows..." part
        valid_times = unit_data.index
        F_r_i = F_r.loc[valid_times].values

        T_i = len(F_r_i)

        if T_i <= R_hat + len(x_cols):
            # Skip units with insufficient obs to estimate
            continue

        # Construct Projection Matrix M_i = I - P_i
        # P_i = F (F'F)^-1 F'
        inv_FiFi = inv(F_r_i.T @ F_r_i)
        P_i = F_r_i @ inv_FiFi @ F_r_i.T
        M_i = np.eye(T_i) - P_i

        # Transform variables
        MX_i = M_i @ X_i
        My_i = M_i @ y_i

        # --- Accumulate for POOLED ---
        denominator_pooled += X_i.T @ MX_i
        numerator_pooled += X_i.T @ My_i

        # --- Calculate for MEAN GROUP ---
        # beta_i = (X' M X)^-1 (X' M y)
        try:
            beta_i = inv(X_i.T @ MX_i) @ (X_i.T @ My_i)
            betas_mg.append(beta_i)
        except np.linalg.LinAlgError:
            pass  # Skip singular matrices

    # Final Pooled Calculation
    beta_pooled = inv(denominator_pooled) @ numerator_pooled

    # Final Mean Group Calculation
    beta_mg = np.mean(betas_mg, axis=0)

    return beta_pooled.flatten(), beta_mg.flatten()

