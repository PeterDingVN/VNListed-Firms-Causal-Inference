from src.utils.scraped_prep import prep_scraped_data, translation
from src.utils.default_libs import *

def col_list(metric):
    if metric == "EBITDA":
        col = ['ebitda_lag1', 'in_stock_lag1', 'invest_nav_lag1', 'long_receive_lag1',
       'long_liability_lag1', 'other_long_asset_lag1', 'cwip_lag1',
       'other_short_asset_lag1', 'long_invest_lag1', 'other_fund_lag1',
       'gov_own_lag1', 'for_own_lag1', 'expense_lag1', 'company', 'year']
        return col

    elif metric == "Value_add":
        col = ['in_stock_lag1', 'invest_nav_lag1', 'long_receive_lag1',
       'long_liability_lag1', 'other_long_asset_lag1', 'cwip_lag1',
       'other_short_asset_lag1', 'long_invest_lag1', 'other_fund_lag1',
       'gov_own_lag1', 'for_own_lag1', 'value_add_lag1', 'company', 'year']
        return col

    elif metric == "Revenue":
        col = ['revenue_lag1', 'in_stock_lag1', 'invest_nav_lag1', 'long_receive_lag1',
       'long_liability_lag1', 'other_long_asset_lag1', 'cwip_lag1',
       'other_short_asset_lag1', 'long_invest_lag1', 'other_fund_lag1',
       'gov_own_lag1', 'for_own_lag1', 'expense_lag1', 'company', 'year']
        return col

    elif metric == "ROA":
        col = ['in_stock_lag1', 'industry', 'other_fund_lag1','for_own_lag1','roa_lag1','loss',
        'liq','oea','gov_own_lag1', 'equity', 'd/e', 'company', 'year']
        return col

    elif metric == "ROE":
        col = ['long_receive_lag1', 'other_fund_lag1', 'roe_lag1',
        'long_invest_lag1', 'in_stock_lag1',
        'asset/equity', 'cash&equi_to_asset', 'a/w',
        'size', 'asset/lia', 'industry','company', 'year']
        return col


def col_trans(path):
    # Raw file
    df = pd.read_excel(path)

    # Trans peformed
    if all("Unnamed" in col for col in df.columns) or any(re.search(r'\b\d{4}', col) for col in df.columns):
        df = prep_scraped_data(path)
    df = translation(df)

    return df










