import pandas as pd
import re

def prep_scraped_data(path):

    '''
    :param path: path to excel file
    :return: dataframe

    Note:
    - this only works with raw excel files with distinctive structure of FiinProX
    - the input file must have one sheet only
    '''

    # Check for sheet number
    try:
        xls = len(pd.ExcelFile(path).sheet_names)
        if xls > 1:
            return ValueError('excel file accepts only one sheet')
    except ValueError:
        return "Only excel file type is accepted"

    # Load data
    try:
        file_ = pd.read_excel(path, header=7)
        idx = file_[file_['STT'].isna()].index.min()
        df = file_[file_.index < idx]
    except Exception:
        df = pd.read_excel(path)

    df_unpivot = df.drop(columns=['STT', 'Tên công ty'])

    # Define id_cols (time invariant columns)
    id_col = []
    for col in df_unpivot.columns:
        if not re.search( r'Q[1-4]\.\d{4}', col):
            id_col.append(col)

    # Unpivot table
    df_unpivot = df_unpivot.melt(id_vars=id_col, value_vars=df_unpivot.drop(columns=id_col).columns)

    # Add date cols
    df_unpivot['datetime'] = df_unpivot['variable'].apply(
      lambda x: re.search( r'Q[1-4]\.\d{4}', x).group()
    )
    df_unpivot[['quar', 'yr']] = df_unpivot['datetime'].str.split('.', expand=True)
    df_unpivot['date'] = df_unpivot['yr'] + df_unpivot['quar']
    df_unpivot['date'] = pd.PeriodIndex(df_unpivot['date'], freq='Q')
    df_unpivot['date'] = df_unpivot['date'].dt.to_timestamp()
    df_unpivot.drop(columns=['datetime', 'yr', 'quar'], inplace=True)


    # clean variable column by adding 'vars_mini' column
    df_unpivot['vars_mini'] = df_unpivot['variable'].apply(
        lambda x: re.sub(r'\d+\.', '', x.split('\n')[0]).strip()
    )
    df_unpivot.drop(columns='variable', inplace=True)

    # pivot columns based on vars_mini
    index = ['date'] + id_col
    df_final = df_unpivot.pivot(
        index=index, columns='vars_mini', values='value'
    ).reset_index()

    return df_final

# df = prep_scraped_data(r'C:\Users\HP\.0_PycharmProjects\Vnlisted_causal\data\cash_flow_FCF_(OPTIONAL).xlsx')
# print(df.shape)
# print(df.columns)
# print(df.head(5))