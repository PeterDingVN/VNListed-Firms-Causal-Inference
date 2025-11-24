from src.utils.default_libs import *
'''

Prep_scraped data is used to preprocess raw data downloaded from FiinProX, so please DO NOT make
any changes to the data scraped from FiinProX before Inserting it into this function
This function then helps reshape the data into proper structure (panel) instead of doing advanced cleaning

'''
def prep_scraped_data(path):

    '''

    :param path: path to excel file
    :return: dataframe

    '''

    # Load data
    try:
        file_ = pd.read_excel(path, header=7)
        idx = file_[file_['STT'].isna()].index.min()
        df = file_[file_.index < idx]
    except:
        df = pd.read_excel(path)

    df_unpivot = df.drop(columns=['STT', 'Tên công ty'])

    # Define id_cols
    id_col = []
    for col in df_unpivot.columns:
        if not re.search(r'\d{4}', col):
            id_col.append(col)

    # Process
    # Unpivot table
    df_unpivot = df_unpivot.melt(id_vars=id_col, value_vars=df_unpivot.drop(columns=id_col).columns)

    # Add year cols
    df_unpivot['year'] = df_unpivot['variable'].apply(
      lambda x: re.search(r'\d{4}', x).group()
    )
    df_unpivot['year'] = pd.to_numeric(df_unpivot['year'], errors='coerce')

    # clean variable column by adding 'vars_mini' column
    df_unpivot['vars_mini'] = df_unpivot['variable'].apply(
        lambda x: re.sub(r'\d+\.', '', x.split('\n')[0]).strip()
    )

    # drop 'variable' column
    df_unpivot.drop(columns='variable', inplace=True)

    # pivot columns based on vars_mini
    index = ['year'] + id_col
    df_final = df_unpivot.pivot(
        index=index, columns='vars_mini', values='value'
    ).reset_index()

    return df_final

# Col translation
'''
This function below helps translate Vietnamese col name into English name if any
'''

def translation(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={
        "Mã": "company",
        "Sàn": "platform",
        "EBITDA": "ebitda",
        "Doanh thu thuần": "revenue",
        "Lợi nhuận thuần từ hoạt động kinh doanh": "net_op_profit",
        "ROA %": "roa",
        "ROE %": "roe",
        "ROIC %": "roic",
        "ROCE %": "roce",
        "Giá vốn hàng bán": "cogs",
        "Chi phí bán hàng": "sales_cost",
        "Chi phí quản lý doanh nghiệp": "admin_cost",
        "Các khoản phải thu ngắn hạn": "short_receive",
        "Tiền và tương đương tiền": "cash",
        "Tài sản ngắn hạn khác": "other_short_asset",
        "Hàng tồn kho ròng": "in_stock",
        "Phải thu dài hạn": "long_receive",
        "Tài sản dài hạn khác": "other_long_asset",
        "Đầu tư dài hạn": "long_invest",
        "Tài sản dở dang dài hạn": "cwip",
        "Tài sản cố định": "fixed_asset",
        "Giá trị ròng tài sản đầu tư": "invest_nav",
        "Nợ dài hạn": "long_liability",
        "Nợ ngắn hạn": "short_liability",
        "Vốn và các quỹ": "equity_fund",
        "Nguồn kinh phí và quỹ khác": "other_fund",
        "Tỷ lệ sở hữu nhà nước": "gov_own",
        "Tỷ lệ sở hữu nước ngoài": "for_own",
        "Phân ngành - ICB L1": "industry",
        "II. VỐN CHỦ SỞ HỮU": "equity",
        "A. TỔNG CỘNG TÀI SẢN": "tot_asset",
        "Vay và nợ thuê tài chính ngắn hạn": "short_debt",
        "Vay và nợ thuê tài chính dài hạn": "long_debt",
        "Số nhân viên hiện tại": "emp_num",
        "I. NỢ PHẢI TRẢ": "lia"
    })

    all_cols = df.columns.tolist()
    if all(col in all_cols for col in ['cogs', 'sales_cost', 'admin_cost']):
        df['expense'] = df['cogs'] + df['sales_cost'] + df['admin_cost']
    elif "net_op_profit" in all_cols:
        df['loss'] = df['net_op_profit'].apply(lambda x: 0 if x <=0 else 1)
    elif all(col in all_cols for col in ['cash', 'tot_asset']):
        df['liq'] = df['cash']/ df['tot_asset']
        df['size'] = np.log(df['tot_asset']+1)
    elif all(col in all_cols for col in ['cogs', 'sales_cost', 'admin_cost', 'tot_asset']):
        df['oea'] = (df['cogs'] + df['sales_cost'] + df['admin_cost'])/df['tot_asset']
    elif all(col in all_cols for col in ['short_debt', 'long_debt', 'equity']):
        df['d/e'] = (df['short_debt']+df['long_debt'])/df['equity']
    elif all(col in all_cols for col in ['equity', 'tot_asset']):
        df['asset/equity'] = df['tot_asset']/df['equity']
    elif all(col in all_cols for col in ['cash', 'equity', 'tot_asset']):
        df['cash&equi_to_asset'] = (df['cash']+df['equity'])/df['tot_asset']
    elif all(col in all_cols for col in ['emp_num', 'tot_asset']):
        df['a/w'] = df['tot_asset']/df['emp_num']
    elif all(col in all_cols for col in ['lia', 'tot_asset']):
        df['asset/lia'] = df['tot_asset']/df['lia']

    return df