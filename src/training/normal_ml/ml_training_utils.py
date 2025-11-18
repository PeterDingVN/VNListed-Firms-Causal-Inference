from src.utils.default_libs import *
from src.training.normal_ml.hyperparams_ML import *
from panelsplit.cross_validation import PanelSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import Tuple

# Class InputData is used to do cross-validation for best algo and params
class InputData:
    def __init__(self, df: pd.DataFrame, id_col: str, time_col: str, target: str, reg: bool):
        '''

        :param df: input dataframe
        :param id_col: cross-sectional identity (exp: company)
        :param time_col: time-series column (exp: year)
        :param target: target/outcome/predicted var
        :param reg: if True, regression metrics will be used for evaluation, else using classification's

        '''

        self.df = df
        self.id_col = id_col
        self.time_col = time_col
        self.target = target
        self.reg = reg

    def optimal_param(self, n_splits: int, test_size: int, gap=1) -> pd.DataFrame:
        '''

        :param n_splits: number of folds to loop through
        :param test_size: limit test set size, max number == n_samples // (n_splits + 1)
        :param gap: number of interval periods between train and val set
        :return: a table summarizing the score of all hyper-parameters from best -> worst

        '''

        # Set index for data input
        idx = [self.id_col, self.time_col]
        df_copy = self.df.copy()
        df_copy.set_index(idx, inplace=True)

        # Define X, y from data input
        X = df_copy.drop([self.target], axis=1)
        y = df_copy[self.target]
        # Define cross-validation method and periods for cv
        periods = df_copy.index.get_level_values(level=1)
        cv_strat = PanelSplit(periods = periods, test_size=test_size, n_splits=n_splits, gap=gap)

        result_fin = []

        # GridSearch CV for regression task
        if self.reg:
            for name, (algo, hyperpar) in algorithm_reg.items():

                # Scaling before regression
                pipe = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('algo', algo)
                ])
                print(f'Processing {name} ...')

                # Gridsearch params
                grid = GridSearchCV(pipe,
                                    scoring=score_reg,
                                    param_grid = hyperpar,
                                    cv=cv_strat,
                                    refit='r2')
                grid_fit = grid.fit(X, y)
                results = pd.DataFrame(grid_fit.cv_results_)
                results['algo_used'] = f'{name}'
                result_fin.append(results[['algo_used', 'params',
                                           'mean_test_r2', 'mean_test_mape', 'mean_test_rmse']])

            # Output result
            eval_output = pd.concat(result_fin, ignore_index=True)
            eval_output.sort_values(by=['mean_test_r2', 'mean_test_rmse', 'mean_test_mape'],
                                    ascending=False,
                                    inplace=True)

        # Same for classification task
        else:
            for name, (algo, hyperpar) in algorithm_class.items():
                pipe = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('algo', algo)
                ])
                print(f'Processing {name} ...')
                grid = GridSearchCV(pipe,
                                    scoring=score_class,
                                    param_grid = hyperpar,
                                    cv=cv_strat,
                                    refit='accuracy')
                grid_fit = grid.fit(X, y)
                results = pd.DataFrame(grid_fit.cv_results_)
                results['algo_used'] = f'{name}'
                result_fin.append(results[['algo_used', 'params',
                                           'mean_test_accuracy',
                                           'mean_test_roc_auc', 'mean_test_precision',
                                           'mean_test_recall']])

            eval_output = pd.concat(result_fin, ignore_index=True)
            eval_output.sort_values(by=['mean_test_accuracy',
                                        'mean_test_roc_auc',
                                        'mean_test_precision',
                                        'mean_test_recall'],
                                    ascending=False,
                                    inplace=True)

        return eval_output


# Function input_test_split helps splitting one data into input and test set
def input_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''

    :param: the full input dataframe
    :return: 2 dataframes divided into input (90%) and hold_out_test set (10%)

    '''

    # Sort data
    all_idx = df['year'].unique().tolist()
    all_idx.sort()

    # Define index
    input_idx = all_idx[:int(len(all_idx) * 0.9)]
    test_idx = all_idx[int(len(all_idx) * 0.9):]

    # From index get data
    df_input = df[df['year'].isin(input_idx)]
    df_test = df[df['year'].isin(test_idx)]

    return df_input, df_test


# Function train_val_split helps split input data into train and val sets
def train_val_split(df: pd.DataFrame, target: str, train_size=0.9) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  
    '''
    :param: the full input dataframe
    :returns: train and validation set
    '''

    # Sort values
    df2 = df.sort_values(by=['company', 'year'])

    # Define train and val index
    years = sorted(df2['year'].unique())
    split = int(len(years) * train_size)
    tr_year, val_year = years[:split], years[split:] 

    # Define X, y based on index
    X = df2.drop(columns=[target]).set_index(['company', 'year'])
    y = df2[['company', 'year', target]].set_index(['company', 'year'])

    # From X, y define Xtr -> y_val
    X_tr = X[X.index.get_level_values('year').isin(tr_year)]
    X_val = X[X.index.get_level_values('year').isin(val_year)]
    y_tr = y[y.index.get_level_values('year').isin(tr_year)]
    y_val = y[y.index.get_level_values('year').isin(val_year)]


    return X_tr, y_tr, X_val, y_val
