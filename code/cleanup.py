import pandas as pd
import numpy as np
from extract_functions import *
def cleanup(companies_saved, limit=None, remove_rows=False):
    if limit == None:
        limit = len(companies_saved) 
    for ticker, availability in list(companies_saved.items())[:limit]:
        print(f"Cleaning {ticker}")
        comp = comp_load(ticker)
        category = get_category(comp.sic)
        if availability[0]:
            frame = pd.read_csv(f"..\companies_data\static\{category}\{ticker}.csv", index_col=0, parse_dates=True)
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\static\{category}\{ticker}.csv")
        if availability[1]:
            frame = pd.read_csv(f"..\companies_data\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\dynamic\{category}\{ticker}.csv")
        if availability[2]:
            frame = pd.read_csv(f"..\companies_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\price\{ticker}.csv")

def quantiles(companies_saved, limit=None, quantile = 0.8, period_dict=None):
    if period_dict == None:
        period_dict = {"static": 30, "dynamic": 90}
    if limit == None:
        limit = len(companies_saved)
    for ticker, availability in list(companies_saved.items())[:limit]:
        if availability[2]:
            print(f"Quantile for {ticker}")
            frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            if availability[0]:
                frame["static_quantile"] = frame['close'].rolling(window=period_dict["static"]).quantile(quantile)
            if availability[1]:
                frame["dynamic_quantile"] = frame['close'].rolling(window=period_dict["dynamic"]).quantile(quantile)
            frame.to_csv(f"..\clean_data\price\{ticker}.csv")

def per_share_divide(companies_saved, limit=None):
    if limit == None:
        limit = len(companies_saved)
    for ticker, availability in list(companies_saved.items())[:limit]:
        if availability[2]:
            print(f"Dividing {ticker}")
            comp = comp_load(ticker)
            category = get_category(comp.sic)
            price_frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            if availability[0]:
                frame = pd.read_csv(f"..\clean_data\static\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                price_frame["EntityCommonStockSharesOutstanding-0"].replace(0, pd.NA, inplace=True)
                frame = frame.divide(price_frame["EntityCommonStockSharesOutstanding-0"], axis=0)
                frame.to_csv(f"..\clean_data\per_share\static\{category}\{ticker}.csv")
            if availability[1]:
                frame = pd.read_csv(f"..\clean_data\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                price_frame = price_frame.reindex(frame.index)
                price_frame["EntityCommonStockSharesOutstanding-0"].replace(0, pd.NA, inplace=True)
                frame = frame.divide(price_frame["EntityCommonStockSharesOutstanding-0"], axis=0)
                frame.to_csv(f"..\clean_data\per_share\dynamic\{category}\{ticker}.csv")  

def ready_data(companies_saved, limit=None):
    if limit == None:
        limit = len(companies_saved) 
    with open("C:\Programming\Python\Finance\EDGAR\categories\category_conversion.json", "r") as file:
        category_conversion = json.load(file)
    for ticker, availability in list(companies_saved.items())[:limit]:
        print(f"Readying {ticker}")
        if availability[2]:
            price_frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            comp = comp_load(ticker)
            category = get_category(comp.sic)
            cat_feature = category_conversion[category]
            if availability[0]:
                static_frame = pd.read_csv(f"..\clean_data\static\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                static_frame = pd.concat([static_frame,price_frame.reindex(static_frame.index)], axis=1)
                static_frame["category"] = cat_feature
                static_frame.to_csv(f"..\\ready_data\static\{ticker}.csv")
            if availability[1]:
                dynamic_frame = pd.read_csv(f"..\clean_data\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                dynamic_frame = pd.concat([dynamic_frame,price_frame.reindex(dynamic_frame.index)], axis=1)
                dynamic_frame["category"] = cat_feature
                dynamic_frame.to_csv(f"..\\ready_data\dynamic\{ticker}.csv")
            if availability[0] and availability[1]:
                total_frame = pd.concat([static_frame,dynamic_frame], axis=1)
                total_frame.to_csv(f"..\\ready_data\\together\{ticker}.csv")
