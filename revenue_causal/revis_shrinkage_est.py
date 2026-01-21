import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class StrictShrinkageEstimator:
    def __init__(self, data, id_col, time_col, y_col, x_cols, max_factors=8, k1=1, k2=1):
        """
        Strict implementation of Lu & Su (2015) for Unbalanced Data.
        """
        self.data = data.sort_values([id_col, time_col]).copy()
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.x_cols = x_cols
        self.R_max = max_factors
        self.k1 = k1
        self.k2 = k2

        self.ids = self.data[id_col].unique()
        self.times = self.data[time_col].unique()
        self.N = len(self.ids)
        self.T = len(self.times)
        self.K = len(x_cols)

        self._prepare_matrices()

    def _prepare_matrices(self):
        # Convert to N x T matrices (with NaNs for missing data)
        self.Y_mat = self.data.pivot(index=self.id_col, columns=self.time_col, values=self.y_col).values

        self.X_mats = []
        for x in self.x_cols:
            mat = self.data.pivot(index=self.id_col, columns=self.time_col, values=x).values
            self.X_mats.append(mat)
        self.X_mats = np.array(self.X_mats)  # Shape (K, N, T)

        # Mask: 1 if observed, 0 if missing
        self.mask = ~np.isnan(self.Y_mat)
        self.total_obs = np.sum(self.mask)

    def _fill_missing(self, matrix):
        # Simple row-mean imputation for PCA steps
        m = matrix.copy()
        # Calculate row means ignoring NaNs
        row_means = np.nanmean(m, axis=1)
        # Find indices where NaN exists
        inds = np.where(np.isnan(m))
        # Replace NaNs with corresponding row mean
        m[inds] = np.take(row_means, inds[0])
        return m

    def _pca_estimation(self, W, R):
        """
        Estimates Factors (F) and Loadings (Lambda) from residuals W via PCA.
        Strictly enforces F'F/T = I.
        """
        W_filled = self._fill_missing(W)

        # PCA on T x T matrix (W'W) to find F
        Sigma = (W_filled.T @ W_filled) / (self.N * self.T)
        evals, evecs = np.linalg.eigh(Sigma)

        # Sort descending
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # F_hat = sqrt(T) * eigenvectors
        F_hat = evecs[:, :R] * np.sqrt(self.T)

        # Lambda_hat = (W * F) / T
        Lambda_hat = (W_filled @ F_hat) / self.T

        return Lambda_hat, F_hat

    def step_1_preliminary(self):
        """
        Step 1: Consistent Initial Estimates
        """
        print("--- Step 1: Preliminary Estimation ---")

        # 1.1 Initial Beta (Iterative Profile QMLE)
        # Flatten data for regression operations
        y_flat = self.Y_mat[self.mask]
        X_flat = np.array([xm[self.mask] for xm in self.X_mats]).T

        beta = np.zeros(self.K)  # Start with 0

        for i in range(50):  # Iterate to converge
            prev_beta = beta.copy()

            # Calculate Residuals W = Y - X*beta
            W = self.Y_mat.copy()
            for k in range(self.K):
                W -= beta[k] * self.X_mats[k]

            # Estimate Factors given Beta
            Lambda, F = self._pca_estimation(W, self.R_max)

            # Update Beta given Factors
            # Effective Y = Y - Lambda*F'
            LF = Lambda @ F.T
            Y_eff = (self.Y_mat - LF)[self.mask]

            # OLS
            if self.K > 0:
                beta = np.linalg.lstsq(X_flat, Y_eff, rcond=None)[0]

            if np.allclose(beta, prev_beta, atol=1e-5):
                break

        beta_tilde = beta

        # 1.2 Bias Correction (CRITICAL STEP)
        print("Warning: Bias Correction set to 0. Real implementation requires Moon & Weidner (2014) HAC estimation.")
        bias_correction_term = np.zeros_like(beta_tilde)
        # To make this rigorous, one must implement the Kernel Density estimation of Long Run Variance here.

        self.beta_c = beta_tilde - bias_correction_term

        # 1.3 Update Factors with clean Beta
        W_c = self.Y_mat.copy()
        for k in range(self.K):
            W_c -= self.beta_c[k] * self.X_mats[k]
        self.lambda_tilde, self.f_tilde = self._pca_estimation(W_c, self.R_max)

        print(f"Beta Corrected (Initial): {self.beta_c}")

    def step_2_weights(self):
        """
        Step 2: Calculate Penalty Weights
        """
        print("--- Step 2: Constructing Weights ---")

        # 1. Regressor Weights (w1)
        # Add small epsilon to handle beta approx 0
        self.w1 = 1.0 / (np.abs(self.beta_c) ** self.k1 + 1e-6)

        # 2. Factor Weights (w2)
        # Calculate proxy factors based on Bias-Corrected Residuals
        W_c = self.Y_mat.copy()
        for k in range(self.K):
            W_c -= self.beta_c[k] * self.X_mats[k]
        W_c = self._fill_missing(W_c)

        # Get eigenvalues of the covariance matrix of proxy factors
        # The paper uses Proxy F_hat = (NT)^-1 Y_hat' Y_hat F_tilde
        # This is equivalent to finding singular values of projected residuals
        Sigma = (W_c.T @ W_c) / self.total_obs
        evals, _ = np.linalg.eigh(Sigma)
        self.tau = np.sort(evals)[::-1][:self.R_max]

        self.w2 = 1.0 / (self.tau ** self.k2 + 1e-6)

        print(f"Weight vector (Regressors): {self.w1}")
        print(f"Weight vector (Factors): {self.w2}")

    def _solve_pl_objective(self, gamma1, gamma2):
        """
        Strict Minimization of Q_gamma(beta, lambda) using Alternating Optimization.
        This follows the article's requirement to minimize the joint function.
        """
        # Initialize
        beta_hat = self.beta_c.copy()
        W = self.Y_mat.copy()
        for k in range(self.K):
            W -= beta_hat[k] * self.X_mats[k]
        Lambda_hat, F_hat = self._pca_estimation(W, self.R_max)

        X_flat = np.array([xm[self.mask] for xm in self.X_mats]).T

        # Alternating Loop
        for iteration in range(20):
            prev_beta = beta_hat.copy()

            # --- A. Update Beta (Fix Lambda, F) ---
            # Problem: min ||(Y - LF') - X*beta||^2 + gamma1 * sum(w1 * |beta|)
            # This is a Weighted Lasso.
            # TRICK: Transform X_k* = X_k / w1_k. Solve standard Lasso. Then beta = beta* / w1_k

            LF = Lambda_hat @ F_hat.T
            Y_target = (self.Y_mat - LF)[self.mask]

            # Scale X by weights
            X_scaled = X_flat / self.w1[None, :]

            # Solve Standard Lasso (sklearn minimizes: (1/2N)||y-Xb||^2 + alpha||b||1)
            # Paper minimizes: (1/NT)||...||^2 + gamma1...
            # We must adjust alpha. sklearn alpha = gamma1 / (NT or N depending on implementation)
            # Sklearn objective is 1/(2 * n_samples) * ||y - Xw||^2_2 + alpha * ||w||_1
            # Our objective is 1/total_obs * ||...||^2 + gamma1 * ||...||
            # So sklearn_alpha = gamma1 / 2.0

            lasso = Lasso(alpha=gamma1 / 2.0, fit_intercept=False, warm_start=True, max_iter=1000)
            lasso.fit(X_scaled, Y_target)

            # Rescale back to get actual beta
            beta_hat = lasso.coef_ / self.w1

            # --- B. Update Factors (Fix Beta) ---
            # Problem: min ||(Y - X*beta) - LF'||^2 + gamma2 * sum(w2 * ||lambda_r||)
            # 1. Compute Residuals W
            W_new = self.Y_mat.copy()
            for k in range(self.K):
                W_new -= beta_hat[k] * self.X_mats[k]

            # 2. PCA to get raw Lambda, F
            Lambda_raw, F_new = self._pca_estimation(W_new, self.R_max)

            # 3. Apply Group Lasso Shrinkage to Lambda columns
            # Formula: lambda_new = lambda_raw * max(0, 1 - penalty / ||lambda_raw||)
            # Penalty per column = (gamma2 * w2_r) / sqrt(N)
            # Note: Paper scales penalty by 1/sqrt(N)

            for r in range(self.R_max):
                col_norm = np.linalg.norm(Lambda_raw[:, r])
                penalty = (gamma2 * self.w2[r]) / np.sqrt(self.N)

                if col_norm <= penalty:
                    scale = 0.0
                else:
                    scale = 1.0 - (penalty / col_norm)

                Lambda_hat[:, r] = Lambda_raw[:, r] * scale

            F_hat = F_new

            # Check Convergence
            if np.allclose(beta_hat, prev_beta, atol=1e-4):
                break

        return beta_hat, Lambda_hat, F_hat

    def step_3_and_4_optimization(self):
        """
        Step 3 & 4: Grid Search using IC
        """
        print("--- Step 3 & 4: Grid Search & Minimization ---")

        # Define Grid (Log-scale is usually better)
        # Note: Values depend on data scale.
        g1_vals = [0.0, 0.01, 0.1, 1.0]
        g2_vals = [0.0, 1.0, 5.0, 10.0]

        results = []

        for g1 in g1_vals:
            for g2 in g2_vals:
                # 1. Solve strictly
                beta_hat, Lambda_hat, F_hat = self._solve_pl_objective(g1, g2)

                # 2. Calculate Effective Parameters
                # Non-zero betas
                S_beta = np.sum(np.abs(beta_hat) > 1e-5)

                # Non-zero factors (columns with norm > 0)
                col_norms = np.linalg.norm(Lambda_hat, axis=0)
                R_hat = np.sum(col_norms > 1e-5)

                # K_eff formula
                K_eff = S_beta + (self.N * R_hat + self.T * R_hat - R_hat ** 2)

                # 3. Calculate MSE (Sigma^2)
                W = self.Y_mat.copy()
                for k in range(self.K):
                    W -= beta_hat[k] * self.X_mats[k]
                LF = Lambda_hat @ F_hat.T
                residuals = (W - LF)[self.mask]  # Use mask for unbalanced
                mse = np.mean(residuals ** 2)

                # 4. Calculate IC
                # C_nt = ln(NT) usually
                C_nt = np.log(self.total_obs)
                ic = np.log(mse) + K_eff * (C_nt / self.total_obs)

                results.append({
                    'gamma1': g1,
                    'gamma2': g2,
                    'ic': ic,
                    'beta': beta_hat,
                    'R_hat': R_hat
                })

        # Find minimum IC
        best_res = min(results, key=lambda x: x['ic'])

        print(f"Optimal Gamma: ({best_res['gamma1']}, {best_res['gamma2']})")
        print(f"Selected R: {best_res['R_hat']}")
        print(f"Final Beta: {best_res['beta']}")

        return best_res['beta'], best_res['R_hat']

    def fit(self):
        self.step_1_preliminary()
        self.step_2_weights()
        return self.step_3_and_4_optimization()


# --- Execution Example ---
if __name__ == "__main__":
    # Create Dummy Unbalanced Data
    np.random.seed(42)
    N, T = 50, 20
    ids = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)

    # True Model: 2 Factors, Beta = [2, 0, -1.5]
    K = 3
    true_beta = np.array([2.0, 0.0, -1.5])

    X = np.random.randn(N * T, K)
    # Generate Factors
    Lam_true = np.random.randn(N, 2)
    F_true = np.random.randn(T, 2)
    LF_true = Lam_true[ids] * F_true[times]
    LF_sum = np.sum(LF_true, axis=1)

    Y = X @ true_beta + LF_sum + np.random.normal(0, 0.5, N * T)

    df = pd.DataFrame({'id': ids, 'time': times, 'y': Y, 'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2]})

    # Make unbalanced: Remove 10% of Y values
    mask = np.random.rand(len(df)) > 0.1
    df.loc[~mask, 'y'] = np.nan

    # Run Estimator
    est = StrictShrinkageEstimator(df, 'id', 'time', 'y', ['x1', 'x2', 'x3'], max_factors=5)
    final_b, final_r = est.fit()