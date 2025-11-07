from utils.utils_dl_train import *
from utils.utils_dta_processing import *
from utils.utils_dl_train.dl_data_preprocess import DataPreprocess

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

