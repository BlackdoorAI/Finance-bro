import pandas as pd
import numpy as np
from extract_functions import *

def adjust_dates(frame):
    # Check if the index is a DatetimeIndex
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    # Vectorized operation to determine if the date is before or after the 15th
    is_before_16 = frame.index.day <= 15

    # Calculate the last day of the previous month
    end_previous_month = frame.index - pd.DateOffset(days=frame.index.day)

    # Calculate the last day of the current month
    start_next_month = pd.to_datetime(frame.index.to_period('M').astype(str)) + pd.DateOffset(months=1)
    end_current_month = start_next_month - pd.DateOffset(days=1)

    # Use numpy where to vectorize the conditional assignment
    adjusted_dates = pd.DatetimeIndex(np.where(is_before_16, end_previous_month, end_current_month))

    frame.index = adjusted_dates
    return frame


def cleanup(companies_saved, limit=None, remove_rows=False, tether_index=False):
    if limit == None:
        limit = len(companies_saved) 
    for ticker, availability in list(companies_saved.items())[:limit]:
        if not (availability[0] or availability[1]) or not availability[2]: #Uninitialized
            continue
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
            if tether_index:
                frame = adjust_dates(frame)
            frame.to_csv(f"..\clean_data\static\{category}\{ticker}.csv")
        if availability[1]:
            frame = pd.read_csv(f"..\companies_data\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            if tether_index:
                frame = adjust_dates(frame)
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
        if availability[2] and (availability[0] or availability[1]):
            print(f"Quantile for {ticker}")
            frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            if availability[0]:
                frame["static_quantile"] = frame['close'].rolling(window=period_dict["static"]).quantile(quantile)
            if availability[1]:
                frame["dynamic_quantile"] = frame['close'].rolling(window=period_dict["dynamic"]).quantile(quantile)
            #Also get the kind of current quantile 
            frame["current_quantile"] = frame['close'].rolling(window=40).quantile(quantile)
            frame.to_csv(f"..\clean_data\price\{ticker}.csv")

def to_list(window):
    return window.tolist()

def upper_averages(companies_saved, limit=None, quantile = 0.8, period_dict=None):
    if period_dict == None:
        period_dict = {"static": 30, "dynamic": 90}
    if limit == None:
        limit = len(companies_saved)
    for ticker, availability in list(companies_saved.items())[:limit]:
        if availability[2] and (availability[0] or availability[1]):
            print(f"Average for {ticker}")
            frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            if availability[0]:
                rolling_df = pd.DataFrame(index=frame.index)
                for i in range(period_dict["static"]):
                    rolling_df[f'lag_{i}'] = frame['close'].shift(i)
                rolling_df = rolling_df.dropna()
                frame['static_average'] = rolling_df.apply(lambda x: x.mean(), axis=1)
                # static_mask = frame['close'].rolling(window=period_dict["static"]).quantile(quantile) < frame['close']
                # frame["static_average"] = frame['close'].where(static_mask).rolling(window=period_dict["static"], min_periods=1).mean()
            if availability[1]:
                rolling_df = pd.DataFrame(index=frame.index)
                for i in range(period_dict["dynamic"]):
                    rolling_df[f'lag_{i}'] = frame['close'].shift(i)
                rolling_df = rolling_df.dropna()
                frame['dynamic_average'] = rolling_df.apply(lambda x: x.mean(), axis=1)
            #Also get the kind of current quantile 
            rolling_df = pd.DataFrame(index=frame.index)
            for i in range(period_dict["static"]):
                rolling_df[f'lag_{i}'] = frame['close'].shift(i)
            rolling_df = rolling_df.dropna()
            frame['current_average'] = rolling_df.apply(lambda x: x.mean(), axis=1)
            frame.to_csv(f"..\clean_data\price\{ticker}.csv")

def per_share_divide(companies_saved, limit=None):
    if limit == None:
        limit = len(companies_saved)
    for ticker, availability in list(companies_saved.items())[:limit]:
        if availability[2] and (availability[0] or availability[1]):
            print(f"Dividing {ticker}")
            comp = comp_load(ticker)
            category = get_category(comp.sic)
            price_frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            shares = price_frame["EntityCommonStockSharesOutstanding-0"].replace(0, pd.NA)
            if availability[0]:
                frame = pd.read_csv(f"..\clean_data\static\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                frame = frame.divide(shares, axis=0)
                frame.to_csv(f"..\clean_data\per_share\static\{category}\{ticker}.csv")
            if availability[1]:
                frame = pd.read_csv(f"..\clean_data\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                shares = shares.reindex(frame.index)
                frame = frame.divide(shares, axis=0)
                frame.to_csv(f"..\clean_data\per_share\dynamic\{category}\{ticker}.csv")  

def categorize(frame, categories, label_name):
    """
    categories = [(-1,0), (0,1)]
    """
    conditions = [frame[label_name] < categories[0][0]] + [(frame[label_name] >= start) & (frame[label_name] <= end) for start,end in categories] + [categories[-1][1] < frame[label_name]]

    column_names = [f'range{i}' for i in range(0,len(categories)+2)]

    for i, condition in enumerate(conditions):
        frame[column_names[i]] = condition.astype(int)

    frame = frame.drop(columns=[label_name])
    return frame

def ready_data(companies_saved, limit=None, per_share =True, dynamic_shift = pd.Timedelta(days=0), static_shift = pd.Timedelta(days=0), multiples=False, marketcap= True, categories = False, averages=False):
    if limit == None:
        limit = len(companies_saved) 
    with open("..\categories\category_conversion.json", "r") as file:
        category_conversion = json.load(file)
    if per_share: #If we are using the data per share we use a different path
        per_share = "\per_share"
    else:
        per_share = ""
    if marketcap:
        size = "MarketCap"
    else:
        size = "EntityCommonStockSharesOutstanding-0"
    for ticker, availability in list(companies_saved.items())[:limit]:
        print(f"Readying {ticker}")
        if availability[2] and (availability[0] or availability[1]):
            price_frame = pd.read_csv(f"..\clean_data\price\{ticker}.csv", index_col=0, parse_dates=True)
            comp = comp_load(ticker)
            category = get_category(comp.sic)
            cat_feature = category_conversion[category]
            #Drop the unncecessary, sometimes the frame does not have static or dynamic
            price_frame["size"] = price_frame[size]
            dropper = ["EntityCommonStockSharesOutstanding-0", "MarketCap", "close", "current_quantile", "current_average"]
            if averages:
                dropper += [i for i in ["dynamic_quantile", "static_quantile"] if i in price_frame.columns]
            else:
                dropper += [i for i in ["dynamic_average", "static_average"] if i in price_frame.columns]
            if averages:
                iterator = [("static_average", static_shift), ("dynamic_average", dynamic_shift)]
            else:
                iterator = [("static_quantile", static_shift), ("dynamic_quantile", dynamic_shift)]
            for label, shift in iterator:
                if label in price_frame.columns:
                    #Shift the quantiles to form labels
                    quant_frame = price_frame[label].copy()
                    quant_frame.index = quant_frame.index - shift
                    if multiples:
                        price_frame[label] = quant_frame.divide(price_frame["current_average" if averages else "current_quantile"], axis=0) -1
                    else:
                        price_frame[label] = quant_frame 
            price_frame.drop(columns=dropper, inplace=True)
            #Add the category as a feature
            price_frame["category"] = cat_feature 
            if availability[0]:
                static_frame = pd.read_csv(f"..\clean_data{per_share}\static\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                #Concat labels and features
                static_frame = pd.concat([static_frame,price_frame.reindex(static_frame.index)], axis=1)
                static_frame.dropna(inplace=True)
                if "dynamic_quantile" in price_frame.columns:
                    static_frame.drop(columns=["dynamic_quantile"],inplace =True)
                if "dynamic_average" in price_frame.columns:
                    static_frame.drop(columns=["dynamic_average"],inplace =True)
                #Categorize if wanted
                if categories:
                    static_frame = categorize(static_frame,[(-1,-0.5), (-0.5, 0), (0,0.1), (0.1,0.2), (0.2,0.3), (0.4,0.5), (0.5,1)], "static_average" if averages else "static_quantile" )
                static_frame.to_csv(f"..\\ready_data\static\{ticker}.csv")
            if availability[1]:
                dynamic_frame = pd.read_csv(f"..\clean_data{per_share}\dynamic\{category}\{ticker}.csv", index_col=0, parse_dates=True)
                #Concat the labels and features
                dynamic_frame = pd.concat([dynamic_frame,price_frame.reindex(dynamic_frame.index)], axis=1)
                dynamic_frame.dropna(inplace=True)
                if "static_quantile" in price_frame.columns:
                    dynamic_frame.drop(columns=["static_quantile"],inplace =True)
                if "static_average" in price_frame.columns:
                    dynamic_frame.drop(columns=["static_average"],inplace =True)
                if categories:
                    dynamic_frame = categorize(dynamic_frame,[(-1,-0.5), (-0.5, 0), (0,0.1), (0.1,0.2), (0.2,0.3), (0.4,0.5), (0.5,1)], "dynamic_average" if averages else "dynamic_quantile")
                dynamic_frame.to_csv(f"..\\ready_data\dynamic\{ticker}.csv")
            if availability[0] and availability[1]:
                total_frame = pd.concat([static_frame,dynamic_frame], axis=1)
                total_frame.to_csv(f"..\\ready_data\\together\{ticker}.csv")
