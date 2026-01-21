import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import svd, inv
import warnings

'''
Assumption:
- In Step 1 PCA to find F and Lambda, simple imputation is conducted to ensure unbalanced data non-NaN structure
- Bias correction in step 2 is not as the original

'''

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ShrinkageEstimator:
    def __init__(self, data, id_col, time_col, y_col, x_cols, max_factors=8, k1=1, k2=1):
        """
        Initialize the estimator with unbalanced panel data.

        Args:
            data: Pandas DataFrame containing the panel data.
            id_col: Name of the individual ID column.
            time_col: Name of the time column.
            y_col: Name of the dependent variable column.
            x_cols: List of independent variable columns.
            max_factors: The maximum number of factors (R) to consider.
            k1, k2: Power parameters for the penalty weights (typically 1 or 2).
        """
        self.data = data.sort_values([id_col, time_col]).copy()
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.x_cols = x_cols
        self.R_max = max_factors
        self.k1 = k1
        self.k2 = k2

        # Get dimensions
        self.ids = self.data[id_col].unique()
        self.times = self.data[time_col].unique()
        self.N = len(self.ids)
        self.T = len(self.times)
        self.K = len(x_cols)

        # Create mapping for matrix conversion
        self._prepare_matrices()

    def _prepare_matrices(self):
        """
        Convert long-format data to N x T matrices, handling unbalanced data with NaNs.
        """
        # Create pivot tables (N x T)
        self.Y_mat = self.data.pivot(index=self.id_col, columns=self.time_col, values=self.y_col).values

        self.X_mats = []
        for x in self.x_cols:
            mat = self.data.pivot(index=self.id_col, columns=self.time_col, values=x).values
            self.X_mats.append(mat)
        self.X_mats = np.array(self.X_mats)  # Shape (K, N, T)

        # Create a mask for observed data (1 if observed, 0 if missing)
        self.mask = ~np.isnan(self.Y_mat)
        self.total_obs = np.sum(self.mask)

    def _fill_missing(self, matrix):
        """Simple imputation for PCA step (replace NaN with row mean) to handle unbalanced nature."""
        # Note: More sophisticated EM algorithms could be used here, but mean-fill is standard for initial PCA
        m = matrix.copy()
        row_means = np.nanmean(m, axis=1)
        inds = np.where(np.isnan(m))
        m[inds] = np.take(row_means, inds[0])
        return m

    def _pca_estimation(self, W, R):
        """
        Perform PCA to estimate Lambda and F.
        W is the residual matrix (N x T).
        """
        # Handle missing values for PCA
        W_filled = self._fill_missing(W)

        # Eigen decomposition of (W'W)/(NT) or similar.
        # Here using SVD on W directly is more numerically stable.
        # W approx Lambda * F'

        # We need F to be T x R and F'F/T = I
        # Calculate WW' (N x N) or W'W (T x T)
        # Strategy: Principal components of W'W

        Sigma = (W_filled.T @ W_filled) / (self.N * self.T)
        evals, evecs = np.linalg.eigh(Sigma)

        # Sort descending
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # F_hat are the eigenvectors corresponding to largest eigenvalues (scaled by sqrt(T))
        F_hat = evecs[:, :R] * np.sqrt(self.T)

        # Lambda_hat = (W * F) / T
        Lambda_hat = (W_filled @ F_hat) / self.T

        return Lambda_hat, F_hat

    def _profile_qmle(self):
        """
        Step 1.2: Calculate Initial QMLE Beta by iterating.
        """
        # Initial guess (OLS ignoring factors)
        # Flatten data for OLS
        y_flat = self.Y_mat[self.mask]
        X_flat = np.array([xm[self.mask] for xm in self.X_mats]).T

        reg = LinearRegression().fit(X_flat, y_flat)
        beta = reg.coef_

        # Iterate to converge
        for i in range(20):  # Max iterations
            prev_beta = beta.copy()

            # Calculate Residuals W
            W = self.Y_mat.copy()
            for k in range(self.K):
                W -= beta[k] * self.X_mats[k]

            # Estimate Factors
            Lambda, F = self._pca_estimation(W, self.R_max)

            # Update Beta: Regress (Y - Lambda*F') on X
            # Effective Y for beta update
            LF = Lambda @ F.T
            Y_eff = (self.Y_mat - LF)[self.mask]

            reg = LinearRegression().fit(X_flat, Y_eff)
            beta = reg.coef_

            if np.allclose(beta, prev_beta, atol=1e-4):
                break

        return beta, Lambda, F

    def _bias_correction(self, beta_tilde, Lambda, F):
        """
        Step 1.4: Bias Correction (Simplified implementation of Moon & Weidner 2014).

        NOTE: A full implementation of Moon & Weidner is extremely complex and requires
        HAC estimation of specific long-run variance components.
        This is a placeholder for the structural operation.
        """
        # Calculate residuals
        W = self.Y_mat.copy()
        for k in range(self.K):
            W -= beta_tilde[k] * self.X_mats[k]
        e_it = W - (Lambda @ F.T)

        # In a real scenario, you calculate B1 and B2 here based on e_it spectral density
        # For this example code, we will assume the bias term is negligible or calculated elsewhere
        # to keep the code runnable without external HAC libraries.

        # Example: bias_term = calculate_hac_bias(e_it, F, Lambda)
        bias_term = np.zeros_like(beta_tilde)

        beta_c = beta_tilde - bias_term
        return beta_c

    def step_1_preliminary(self):
        print("--- Step 1: Preliminary Estimation ---")
        # 1. Initial QMLE
        beta_tilde, lambda_tilde, f_tilde = self._profile_qmle()

        # 2. Bias Correction
        self.beta_c = self._bias_correction(beta_tilde, lambda_tilde, f_tilde)

        # 3. Re-estimate factors with clean beta
        W_c = self.Y_mat.copy()
        for k in range(self.K):
            W_c -= self.beta_c[k] * self.X_mats[k]

        self.lambda_tilde, self.f_tilde = self._pca_estimation(W_c, self.R_max)

        print(f"Beta Corrected: {self.beta_c}")
        return self.beta_c, self.lambda_tilde, self.f_tilde

    def step_2_weights(self):
        print("--- Step 2: Constructing Weights ---")
        # 1. Regressor Weights
        # Add small epsilon to avoid division by zero
        self.w1 = 1.0 / (np.abs(self.beta_c) ** self.k1 + 1e-8)

        # 2. Proxy Factor Matrix
        W_c = self.Y_mat.copy()
        for k in range(self.K):
            W_c -= self.beta_c[k] * self.X_mats[k]

        # Fill missing for matrix multiplication
        W_c_filled = self._fill_missing(W_c)

        # F_hat = (NT)^-1 * Y_hat' * Y_hat * F_tilde (Simplification from paper logic)
        # Actually paper says F_hat comes from PCA on Y_hat, let's follow the eigen logic
        # Calculating covariance of the factors proxy

        # Recalculate PCA on W_c to get eigenvalues
        Sigma = (W_c_filled.T @ W_c_filled) / self.total_obs
        evals, _ = np.linalg.eigh(Sigma)

        # Sort eigenvalues (largest to smallest)
        self.tau = np.sort(evals)[::-1][:self.R_max]

        # 3. Factor Weights
        self.w2 = 1.0 / (self.tau ** self.k2 + 1e-8)

        print(f"Weights Beta: {self.w1}")
        print(f"Weights Factors (first 3): {self.w2[:3]}...")

    def _objective_function(self, beta, gamma1, gamma2):
        """
        Calculate the PLS objective function value.
        Optimization is done iteratively (coordinate descent logic).
        """
        # 1. Loss Function (SSE)
        # We need to re-estimate Lambda for *this specific* beta to get SSE
        W = self.Y_mat.copy()
        for k in range(self.K):
            W -= beta[k] * self.X_mats[k]

        # For the PLS, factors are often treated as fixed from Step 2,
        # OR re-estimated. The paper minimizes over Beta AND Lambda.
        # Efficient approach: Profile out Lambda.
        Lambda_est, F_est = self._pca_estimation(W, self.R_max)
        LF = Lambda_est @ F_est.T
        residuals = (W - LF)

        # Handle unbalanced SSE
        sse = np.nansum(residuals ** 2) / self.total_obs

        # 2. Penalty 1 (Beta)
        pen1 = gamma1 * np.sum(self.w1 * np.abs(beta))

        # 3. Penalty 2 (Factors)
        # The group lasso penalty on Lambda columns.
        # Since we just re-estimated Lambda_est via PCA, we apply penalty to its columns.
        norms = np.linalg.norm(Lambda_est, axis=0)  # L2 norm of columns
        pen2 = (gamma2 / np.sqrt(self.N)) * np.sum(self.w2 * norms)

        return sse + pen1 + pen2, sse, norms

    def step_3_and_4_optimization(self):
        print("--- Step 3 & 4: Grid Search & Minimization ---")

        # Define Grid (Simplified for demonstration)
        g1_vals = [0, 0.1, 1.0, 10.0]
        g2_vals = [0, 0.1, 1.0, 10.0]

        best_ic = float('inf')
        best_params = None
        best_beta = None
        best_R = None

        # Grid Search
        for g1 in g1_vals:
            for g2 in g2_vals:
                # Solve PLS for this pair (Iterative Coordinate Descent)
                # Initialize with beta_c
                beta_hat = self.beta_c.copy()

                # Simple Soft-Thresholding update for Beta (fixing factors)
                # In real code, use an optimization solver like L-BFGS-B or coordinate descent
                # Here we simulate the result of shrinkage:

                # Apply Thresholding logic roughly:
                # If signal < penalty, set to 0
                for k in range(self.K):
                    threshold = g1 * self.w1[k]
                    if abs(beta_hat[k]) < threshold:
                        beta_hat[k] = 0
                    else:
                        # Shrink towards zero
                        sign = np.sign(beta_hat[k])
                        beta_hat[k] = sign * (abs(beta_hat[k]) - threshold)

                # Calculate Factor norms and shrink
                # We need the Lambda corresponding to this Beta
                W = self.Y_mat.copy()
                for k in range(self.K):
                    W -= beta_hat[k] * self.X_mats[k]
                Lambda_temp, _ = self._pca_estimation(W, self.R_max)
                col_norms = np.linalg.norm(Lambda_temp, axis=0)

                # Count non-zero factors (Group Lasso Logic)
                # If norm < penalty, factor is removed
                factor_threshold = (g2 / np.sqrt(self.N)) * self.w2
                active_factors = np.sum(col_norms > factor_threshold)

                # Calculate IC
                # Re-calculate MSE with shrunken beta and active factors
                resid_sq_sum = 0  # Placeholder for exact SSE after shrinkage

                # Effective parameters
                S_beta = np.sum(beta_hat != 0)
                # df for factors approx N*R + T*R - R^2
                df_factors = self.N * active_factors + self.T * active_factors - active_factors ** 2
                K_eff = S_beta + df_factors

                # Calculate Sigma^2 (MSE)
                # Note: In practice, re-run OLS with selected variables/factors to get MSE
                mse = self._objective_function(beta_hat, 0, 0)[1]  # Get raw SSE

                # IC = ln(MSE) + K_eff * C_nt / NT
                C_nt = np.log(self.total_obs)  # Example penalty
                ic = np.log(mse) + K_eff * (C_nt / self.total_obs)

                if ic < best_ic:
                    best_ic = ic
                    best_params = (g1, g2)
                    best_beta = beta_hat
                    best_R = active_factors

        print(f"Optimal Gamma: {best_params}")
        print(f"Selected Factors: {best_R}")
        print(f"Final Sparse Beta: {best_beta}")

        return best_beta, best_R

    def fit(self):
        self.step_1_preliminary()
        self.step_2_weights()
        beta, r = self.step_3_and_4_optimization()
        return beta, r


# --- Example Usage ---
# Create dummy unbalanced data
data = {
    'id': np.repeat(np.arange(10), 20),
    'time': np.tile(np.arange(20), 10),
    'y': np.random.randn(200),
    'x1': np.random.randn(200),
    'x2': np.random.randn(200),  # Irrelevant
    'x3': np.random.randn(200)
}
df = pd.DataFrame(data)

# Introduce unbalance (missing values)
df.loc[df.sample(20).index, 'y'] = np.nan

estimator = ShrinkageEstimator(df, 'id', 'time', 'y', ['x1', 'x2', 'x3'], max_factors=4)
final_beta, final_factors = estimator.fit()