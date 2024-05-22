import pandas as pd
import numpy as np
from extract_functions import *
def cleanup(companies_saved, limit=None, remove_rows=False):
    if limit == None:
        limit = len(companies_saved) 
    for ticker, availability in list(companies_saved.items())[:limit]:
        comp = comp_load(ticker)
        category = get_category(comp.sic)
        if availability[0]:
            frame = pd.read_csv(f"..\companies_data\static\{category}\{ticker}.csv")
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\static\{category}\{ticker}.csv")
        if availability[1]:
            frame = pd.read_csv(f"..\companies_data\dynamic\{category}\{ticker}.csv")
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\dynamic\{category}\{ticker}.csv")
        if availability[2]:
            frame = pd.read_csv(f"..\companies_data\price\{ticker}.csv")
            if remove_rows:
                frame.dropna()
            else:
                mask = frame.isna().any(axis=1)
                frame[mask] = np.nan
            frame.to_csv(f"..\clean_data\price\{ticker}.csv")