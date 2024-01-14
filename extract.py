from requests import get
from json import load,dump
import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime

def getcik(ticker):
    #Convert the ticker into the proper cik
    for key,value in cikdata.items():
        if value["ticker"] == ticker:
            cik = value["cik_str"]
            break
    return str(cik).zfill(10)

#Headers for EDGAR call
headers = {
    "User-Agent":"ficakc@seznam.cz",
    "Accept-Encoding":"gzip, deflate",
}

# cik_url =  "https://www.sec.gov/files/company_tickers.json"
# cikdata = requests.get(cik_url, headers=headers).json()

with open(r"C:\Programming\Python\Finance\EDGAR\cik.json","r") as file:
    cikdata = load(file)
    file.close()
with open(r"C:\Programming\Python\Finance\EDGAR\apple.json","r") as file:
    Apple = load(file)
    file.close()


def companyfacts(ticker):
    #Get all the financial data for a ticker
    cik = getcik(ticker)
    data_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = get(data_url, headers = headers)
    return facts

def endtodatetime(dataframe):
    dataframe.loc[:,"end"] = pd.datetime(dataframe["end"])
    return dataframe

class Stock:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.cik = getcik(self.ticker)
        self.data = companyfacts(self.ticker).json()
        self.start_year = min([datetime.strptime(self.data["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["end"], r"%Y-%m-%d"),datetime.strptime(self.data["facts"]["dei"]["EntityPublicFloat"]["units"]["USD"][0]["end"], r'%Y-%m-%d'),datetime.strptime(self.data["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["end"], r'%Y-%m-%d')])
        self.end_year = datetime.now().date()
        self.fullprice = si.get_data(self.ticker,self.start_year, self.end_year).reset_index()
        self.price = self.fullprice[[self.fullprice.columns[0], "adjclose"]]
    def fact(self,measure,simple=True):
        try:
            point_list = self.data["facts"]["us-gaap"][measure]["units"]["USD"]
            frame = pd.DataFrame(point_list)
            frame[measure] = frame["val"]
            if simple:
                return frame[["end", measure]]
            return frame
        except KeyError:
            print(f"Measure {measure} not available for company.")
    def shares(self,simple=True):
        try:
            if simple:
                return pd.DataFrame(self.data["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["USD"])[["end","val"]]
            return pd.DataFrame(self.data["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["USD"])
        except KeyError:
            if simple:
                return pd.DataFrame(self.data["facts"]["dei"]["EntityPublicFloat"]["units"]["USD"])[["end","val"]]
            return pd.DataFrame(self.data["facts"]["dei"]["EntityPublicFloat"]["units"]["USD"])


Apple = Stock("aapl")
measures = ["Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent"]
stock = Apple
shares = stock.shares().copy()
stock_num = stock.price.copy()
stock_num["end"] = stock_num["index"].astype(str)
stock_num.drop(columns =["index"],inplace=True)
df = pd.merge(shares, stock_num, left_on=["end"], right_on=["end"], how = "left")
df.head(10)