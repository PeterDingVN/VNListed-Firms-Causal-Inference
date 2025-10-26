from . import *
### ----------------------------------
# Below are function used ONLY WHEN you first downloaded data from FinnProX
# You can opt to preprocess the data on Excel
# Note that: 'STT', 'Mã', 'Tên công ty', 'Sàn' is default columns of FiinProX raw dataset
### -----------------------------------

def prep_scraped_data(path,
              sheet_name: str ='Sheet1',
              time_invariant: list = None):

    '''

    :param path: path to excel file
    :param sheet_name: sheet name
    :param time_invariant: columns that are not melted during unpivot, default to ['Mã', 'Sàn'].
                 Add columns that are not time variant. Below are examples of cols with and without time
                 A. TỔNG CỘNG TÀI SẢN Hợp nhất Quý: Hàng năm Năm: 2015 Đơn vị: VND (2015 is year factor)
                 Số nhân viên hiện tại, Phân ngành (No year factor)
    :return: dataframe

    '''
    if time_invariant is None:
        id_col = ['Mã', 'Sàn']
    else:
        id_col = ['Mã', 'Sàn'] + time_invariant

    # Select area with relevant data
    file_ = pd.read_excel(path, sheet_name=sheet_name, header=7)
    idx = file_[file_['STT'].isna()].index.min()
    df = file_[file_.index < idx]

    # drop unnecessary cols
    df_unpivot = df.drop(columns=['STT', 'Tên công ty'])

    # Process
    # Unpivot table
    df_unpivot = df_unpivot.melt(id_vars=id_col, value_vars=df_unpivot.drop(columns=id_col).columns)

    # Add year cols
    df_unpivot['year'] = df_unpivot['variable'].apply(
      lambda x: re.search(r'\d{4}', x).group()
    )
    df_unpivot['year'] = pd.to_numeric(df_unpivot['year'], errors='coerce')



    # clean variables column by adding 'vars_mini' column
    df_unpivot['vars_mini'] = df_unpivot['variable'].apply(
      lambda x: x.split('\n')[0]
    )
    df_unpivot['vars_mini'] = df_unpivot['vars_mini'].apply(
      lambda x: re.sub(r'\d+\.', '', x)
    )
    df_unpivot['vars_mini'] = df_unpivot['vars_mini'].apply(
      lambda x: x.strip()
    )
    # drop 'variable' column
    df_unpivot.drop(columns='variable', inplace=True)

    # pivot columns based on vars_mini
    index = ['year'] + id_col
    df_final = df_unpivot.pivot(
        index=index, columns='vars_mini', values='value'
    ).reset_index()

    return df_final

# Save any data as .CSV
def save_file(df_, name):
  df_.to_csv(f"{name}.csv", index=False)