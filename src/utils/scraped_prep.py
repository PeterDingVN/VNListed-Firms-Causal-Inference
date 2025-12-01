from src.utils.default_libs import *
'''
Prep_scraped data is used to preprocess raw data downloaded from FiinProX, so please DO NOT make
any changes to the data scraped from FiinProX before Inserting it into this function.
This function then helps reshape the data into proper structure (panel) not advanced cleaning.
'''

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
    except:
        df = pd.read_excel(path)

    df_unpivot = df.drop(columns=['STT', 'Tên công ty'])

    # Define id_cols
    id_col = []
    for col in df_unpivot.columns:
        if not re.search(r'\b\d{4}', col):
            id_col.append(col)

    # Unpivot table
    df_unpivot = df_unpivot.melt(id_vars=id_col, value_vars=df_unpivot.drop(columns=id_col).columns)

    # Add year cols
    df_unpivot['year'] = df_unpivot['variable'].apply(
      lambda x: re.search(r'\b\d{4}', x).group()
    )
    df_unpivot['year'] = pd.to_numeric(df_unpivot['year'], errors='coerce')

    # clean variable column by adding 'vars_mini' column
    df_unpivot['vars_mini'] = df_unpivot['variable'].apply(
        lambda x: re.sub(r'\d+\.', '', x.split('\n')[0]).strip()
    )
    df_unpivot.drop(columns='variable', inplace=True)

    # pivot columns based on vars_mini
    index = ['year'] + id_col
    df_final = df_unpivot.pivot(
        index=index, columns='vars_mini', values='value'
    ).reset_index()

    return df_final

'''

"translation" helps translate Vietnamese col name into English name if any

'''

def translation(df: pd.DataFrame):
    '''

    Args:
        df: input the table with VNese col names
    Returns:
        df: return table with VNese cols translated into English

    '''
    df = df.copy()

    # Translation dict
    df = df.rename(columns={
        "Mã": "company",
        "Sàn": "platform",
        "EBITDA": "ebitda_lag1",
        "Doanh thu thuần": "revenue_lag1",
        "Lợi nhuận thuần từ hoạt động kinh doanh": "net_op_profit",
        "ROA %": "roa_lag1",
        "ROE %": "roe_lag1",
        "ROIC %": "roic",
        "ROCE %": "roce",
        "Giá vốn hàng bán": "cogs",
        "Chi phí bán hàng": "sales_cost",
        "Chi phí quản lý doanh  nghiệp": "admin_cost",  # typo: redundant space
        "Chi phí quản lý doanh nghiệp": "admin_cost",   # This ensure smooth running when Fiinpro fixes
        "Các khoản phải thu ngắn hạn": "short_receive_lag1",
        "Tiền và tương đương tiền": "cash",
        "Tài sản ngắn hạn khác": "other_short_asset_lag1",
        "Hàng tồn kho, ròng": "in_stock_lag1",
        "Phải thu dài hạn": "long_receive_lag1",
        "Tài sản dài hạn khác": "other_long_asset_lag1",
        "Đầu tư dài hạn": "long_invest_lag1",
        "Tài sản dở dang dài hạn": "cwip_lag1",
        "Tài sản cố định": "fixed_asset",
        "Giá trị ròng tài sản đầu tư": "invest_nav_lag1",
        "Nợ dài hạn": "long_liability_lag1",
        "Nợ ngắn hạn": "short_liability_lag1",
        "Vốn và các quỹ": "equity_fund",
        "Nguồn kinh phí và quỹ khác": "other_fund_lag1",
        "Tỷ lệ sở hữu nhà nước": "gov_own_lag1",  # Dummy var
        "Tỷ lệ sở hữu nước ngoài": "for_own_lag1",  # dummy var
        "Phân ngành - ICB L1": "industry",
        "II. VỐN CHỦ SỞ HỮU": "equity",
        "A. TỔNG CỘNG TÀI SẢN": "tot_asset",
        "Vay và nợ thuê tài chính ngắn hạn": "short_debt",
        "Vay và nợ thuê tài chính dài hạn": "long_debt",
        "Số nhân viên hiện tại": "emp_num",
        "I. NỢ PHẢI TRẢ": "lia"
    })

    # Extended translation in the existence of prticular col combos
    all_cols = df.columns.tolist()
    if all(col in all_cols for col in ['cogs', 'sales_cost', 'admin_cost']):
        df['expense_lag1'] = df['cogs'] + df['sales_cost'] + df['admin_cost']
    if all(col in all_cols for col in ['revenue_lag1', 'cogs', 'sales_cost', 'admin_cost']):
        df['value_add_lag1'] = df['revenue_lag1'] - (df['cogs'] + df['sales_cost'] + df['admin_cost'])
    if "net_op_profit" in all_cols:
        df['loss'] = df['net_op_profit'].apply(lambda x: 0 if x <=0 else 1)
    if all(col in all_cols for col in ['cash', 'tot_asset']):
        df['liq'] = df['cash']/ df['tot_asset']
        df['size'] = np.log(df['tot_asset']+1)
    if all(col in all_cols for col in ['cogs', 'sales_cost', 'admin_cost', 'tot_asset']):
        df['oea'] = (df['cogs'] + df['sales_cost'] + df['admin_cost'])/df['tot_asset']
    if all(col in all_cols for col in ['short_debt', 'long_debt', 'equity']):
        df['d/e'] = (df['short_debt']+df['long_debt'])/df['equity']
    if all(col in all_cols for col in ['equity', 'tot_asset']):
        df['asset/equity'] = df['tot_asset']/df['equity']
    if all(col in all_cols for col in ['cash', 'equity', 'tot_asset']):
        df['cash&equi_to_asset'] = (df['cash']+df['equity'])/df['tot_asset']
    if all(col in all_cols for col in ['emp_num', 'tot_asset']):
        df['a/w'] = df['tot_asset']/df['emp_num']
    if all(col in all_cols for col in ['lia', 'tot_asset']):
        df['asset/lia'] = df['tot_asset']/df['lia']
    if 'gov_own_lag1' in all_cols:
        df['gov_own_lag1'] = df['gov_own_lag1'].notna().astype(int)
    if 'for_own_lag1' in all_cols:
        df['for_own_lag1'] = df['for_own_lag1'].notna().astype(int)


    return df