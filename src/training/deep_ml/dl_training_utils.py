from src.utils import *
from src.training.deep_ml import *

# -------------------------------------- DataPreprocess class -------------------------------------
# DataPreprocess will:
# split input into train and val set
# apply tensor on data with shape (rows, timestep, features) to fit with LSTM
# scale data to better LSTM perf
class DataPreprocess:
    def __init__(self,
                 df_input: pd.DataFrame,
                 df_test: pd.DataFrame,
                 target: str,
                 features: str | list[str],
                 timestep: int = 1,
                 val_size: float = 0.3):

        # Condition to align val_size with timestep
        ## 1: val set size must be at least 1 year
        num_year = int(len(df_input['year'].unique()) * val_size)
        if num_year <= 0 or val_size > 1 - val_size:
            min_size = float(1 / int(len(df_input['year'].unique())))
            raise ValueError(f"Max val size is 0.5 and Min val_size is {min_size}")

        ## 2: timestep must be smaller than val year
        if timestep <= 0 or timestep > int(len(df_input['year'].unique()) * val_size):
            max_time = int(len(df_input['year'].unique()) * val_size)
            raise ValueError(f"Min timestep is 1 and max is {max_time}")
        elif timestep > len(df_test['year'].unique()):
            max_ = len(df_test['year'].unique())
            raise ValueError(f"Max timestep is {max_}")

        # Class given params
        self.timestep = timestep
        self.val_size = val_size
        self.features = (features if isinstance(features, list) else [features])
        self.target = target

        # Created data
        ## Specify columns
        self.all_cols = ['company', 'year'] + self.features + [self.target]
        df_input = df_input.reset_index()[self.all_cols]
        df_test = df_test.reset_index()[self.all_cols]

        ## Preprocess data
        ### Define test data
        self.test = df_test.sort_values(['company', 'year']).set_index(['company', 'year'])

        ### Split input into train and val set
        self.train, self.val = self.train_validation_split(df_input, self.val_size)


        ## Final scaled data
        all_data_scaled = self.scaler([self.train, self.val, self.test])

        self.X_train_scaled, self.y_train = self.create_timestep(all_data_scaled[0])
        self.X_val_scaled, self.y_val = self.create_timestep(all_data_scaled[1])
        self.X_test_scaled, self.y_test = self.create_timestep(all_data_scaled[2])

    # Function that split df_input -> train and val
    def train_validation_split(self, df: pd.DataFrame, val_size: float = 0.3):
        ## Sort
        all_idx = df['year'].unique()
        all_idx.sort()

        ## Define idx
        lim = int((1 - val_size) * len(all_idx))
        train_idx = all_idx[:lim]
        val_idx = all_idx[lim:]

        ## Final data
        train_df = df[df['year'].isin(train_idx)]
        val_df = df[df['year'].isin(val_idx)]

        ## Set index for final data
        train_df = train_df.sort_values(['company', 'year']).set_index(['company', 'year'])
        val_df = val_df.sort_values(['company', 'year']).set_index(['company', 'year'])

        return train_df, val_df

    # Scale data using PolyFeatures, OnehotEncode, StandardScaler
    def scaler(self, df_scaled: pd.DataFrame | list[pd.DataFrame]):

        ## Ensure format
        df_scaled = (df_scaled if isinstance(df_scaled, list) else [df_scaled])

        ## Data to be fitted scaler on
        fit_data = self.train

        cat_col = fit_data.select_dtypes(exclude='number').columns
        num_col = fit_data.drop(columns=self.target).select_dtypes(include='number').columns

        ## All Scaler squeezed into pipeline
        num_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        col_trans = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_col),
                ("cat", cat_pipe, cat_col)
            ],
            remainder="passthrough",
            sparse_threshold=0
        )

        ## Returned full list of scaled datasets
        col_trans.fit(fit_data)
        all_data = []
        for data in df_scaled:

            data_scaled = col_trans.transform(data)
            cols = col_trans.get_feature_names_out()
            data_scaled = pd.DataFrame(data_scaled, columns=cols, index=data.index)
            if self.target not in data_scaled.columns:
                data_scaled = data_scaled.rename(columns={f'remainder__{self.target}': self.target})
            all_data.append(data_scaled)

        return all_data


    # Create tensor (obs, timestep, features)
    def create_timestep(self, df_: pd.DataFrame):
        df = df_.copy()

        ## Ensure format of input data
        if not isinstance(self.target, str):
            raise ValueError("target must be a string")

        if df.index.names != ['company', 'year']:
            df = df.sort_values(['company', 'year']).set_index(['company', 'year'])

        ## Main function
        X, y = [], []
        for idx, grp in df.groupby('company'):

            if self.timestep > len(grp):
                warnings.warn(f"Some company with year data less than {self.timestep} will be dropped")
                continue

            for i in range(self.timestep, len(grp) + 1):
                X_inp = grp.drop(columns=self.target).iloc[i - self.timestep:i]
                y_inp = grp[self.target].iloc[i - 1]

                X.append(X_inp.values)
                y.append(y_inp)

        return np.array(X), np.array(y).reshape(-1)

# --------------------------------------- LSTM CLASS -----------------------------------------
# LSTM will:
# Run LSTM model on data from DataPreprocess
# Summerize result of train, val, test set
class LSTM(DataPreprocess):
    def __init__(self,
                 df_input: pd.DataFrame,
                 df_test: pd.DataFrame,
                 target: str,
                 features: str | list[str],
                 model_name: str,
                 timestep: int = 1,
                 val_size: float = 0.15,
                 epochs: int = 150,
                 batch_size: int = 32,
                 patience: int = 10,
                 verbose: int = 0
                 ):
        # All params
        ## From DataPreprocess
        super().__init__(df_input, df_test, target, features, timestep, val_size)
        self.fea_len = len(features)

        ## From local
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = callbacks.EarlyStopping(patience=patience,
                                                mode='min',
                                                restore_best_weights=True,
                                                start_from_epoch=10)
        self.verbose = verbose

        ## Predicted result
        self.y_train_hat = self.LSTM_model(model_name).predict(self.X_train_scaled).squeeze()
        self.y_val_hat = self.LSTM_model(model_name).predict(self.X_val_scaled).squeeze()
        self.y_test_hat = self.LSTM_model(model_name).predict(self.X_test_scaled).squeeze()

    # Function to calculate Symmetric Mean Abs Perc Error
    @staticmethod
    def smape(y_true, y_pred):
        ## Dimension conversion 3d -> 2d
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        ## Formula
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2

        # Ensure no div/0 error
        denom[denom == 0] = 1
        return float(np.mean(np.abs((y_true - y_pred) / denom)))

    # Build LSTM model
    def LSTM_model(self, model_name: str):
        if model_name == 'roa':
            model = Sequential([
                layers.Input(shape=self.X_train_scaled.shape[1:]),
                layers.Dense(16),
                layers.LSTM(32, return_sequences=True),
                layers.LSTM(64,  return_sequences=True),
                layers.LSTM(64, return_sequences=False),
                layers.Dense(16),
                layers.Dense(1)
            ])

        elif model_name == 'roe':
            model = Sequential([
                layers.Input(shape=self.X_train_scaled.shape[1:]),
                layers.LSTM(16, return_sequences=True),
                layers.LSTM(16,  return_sequences=True),
                layers.LSTM(32, return_sequences=False),
                layers.Dense(32),
                layers.Dense(1)
            ])

        else:
            raise ValueError('Available model_names are: roa, roe')

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss='mae'
        )

        model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.patience],
            verbose=self.verbose
        )

        return model

    # Summerize result
    def result_summary(self):
        ## R2
        train_r2 = r2(self.y_train, self.y_train_hat)
        val_r2 = r2(self.y_val, self.y_val_hat)
        test_r2 = r2(self.y_test, self.y_test_hat)

        ## MAE
        train_mae = mae(self.y_train, self.y_train_hat)
        val_mae = mae(self.y_val, self.y_val_hat)
        test_mae = mae(self.y_test, self.y_test_hat)

        ## SMAPE
        train_smape = LSTM.smape(self.y_train, self.y_train_hat)
        val_smape = LSTM.smape(self.y_val, self.y_val_hat)
        test_smape = LSTM.smape(self.y_test, self.y_test_hat)

        # Result output
        result = pd.DataFrame({
            'R2': [train_r2, val_r2, test_r2],
            'MAE': [train_mae, val_mae, test_mae],
            'SMAPE': [train_smape, val_smape, test_smape]
        }, index=['Train', 'Val', 'Test'])

        return result

