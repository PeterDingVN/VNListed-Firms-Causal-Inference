# from src.utils.scraped_prep import prep_scraped_data, translation
# from src.utils.default_libs import *
#
# def load_data(path):
#     if not path.lower().endswith(('.xlsx', '.csv')):
#         raise ValueError('mlflow only works with .xlsx and .csv files')
#
#     try:
#         df = pd.read_excel(path)
#         if all("Unnamed" in col for col in df.columns) or any(re.search(r'Năm:\s*\d{4}', col) for col in df.columns):
#             df = prep_scraped_data(path)
#     except:
#         df = pd.read_csv(path)
#
#     df = translation(df)
#     return df
#
# # Kcan lag vi ai cx bt forecast Y+1
# # Preprocess data input
# # Predict









