import httpx
from fredapi import Fred
import numpy as np
import json
import pandas as pd
import pickle
# We use normal datetime for fred info and pandas datetime for data
from datetime import datetime
import math
import os
from functools import lru_cache
from conversions import *
from net_stuff import *
from reshape import *
from collections import defaultdict


FACTS_PATH = r"C:\Edgar_data"
SUBMISSIONS_PATH = r"C:\Submissions_data"

fred = Fred(api_key='0c34c4dd2fd6943f6549f1c990a8a0f0') 
TIMEOUT = 8
RETRIES = 2
START_RETRY_DELAY = 3
#Easily load a company into a variable

with open(r'..\other_pickle\unavailable.json', 'r') as file:
        Unavailable_Measures = json.load(file)

#Lookup table for the undeprecated version of a measure
with open(r"..\other_pickle\deprecated_to_current.json", "r") as file:
    deprecate_conversion = json.load(file)
    file.close()

#Categories
with open(r"..\categories\categories.json", "r") as file:
    categories = json.load(file)

#Irrelevants
with open(r"..\categories\category_measures_irrelevant.json") as file:
    irrelevants = json.load(file)
with open(r"..\categories\category_measures.json", "r") as file:
    category_measures = json.load(file)
#The first entry date into the EDGAR database
START = datetime.strptime('1993-01-01', r"%Y-%m-%d")

#Headers for EDGAR call
headers = {
    "User-Agent":"ficakc@seznam.cz",
    "Accept-Encoding":"gzip, deflate",
}

with open(r"..\other_pickle\cik.json","r") as file:
    cikdata = json.load(file)
    file.close()

def comp_load(ticker):
    try:
        with open(f"C:\Programming\Python\Finance\EDGAR\companies\{ticker}.pkl", "rb") as file: #The path here is beacause it's meant to run in extract
            company = pickle.load(file)
    except FileNotFoundError:
        print("File doesn't exist")
        return None
    return company

def example_save(data, name):
    with open(f"C:\Programming\Python\Finance\EDGAR\examples/{name}.json", "w") as file: #The path here is beacause it's meant to run in extract
        json.dump(data,file,indent=1)
    print("Saved")
    
#Manually figure out which measure is used with some company
def company_wordsearch(ticker, word):
    with open(f"..\\companies\{ticker}.pkl", "rb")as file:
        company = pickle.load(file)
    data = company.data
    compdict = {}
    for key,value in data.items():
        compdict[key] = value["description"]

    matching_elements  ={}
    for name, desc in compdict.items():
        if word.lower() in name.lower():   
            matching_elements[name] = desc 
    with open(f"..\\checkout\{ticker}.json", "w") as file:
        json.dump(matching_elements, file, indent =1)
    formatted_json = json.dumps(matching_elements, indent=4)
    formatted_with_newlines = formatted_json.replace('\n', '\n\n')
    print(formatted_with_newlines)  

@lru_cache(maxsize=None)
def closest_date(dates, target_date, ticker, fallback=False):
    left, right = 0, len(dates) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if dates[mid] < target_date:
            left = mid + 1
        elif dates[mid] > target_date:
            right = mid - 1
        else:
            # Exact match
            if fallback:
                # Ensure mid-1 is within bounds
                return dates[mid], dates[mid-1] if mid-1 >= 0 else None
            return dates[mid]
    
    if left > 0:
        if fallback:
            # Ensure left-2 is within bounds for the second closest date
            second_closest = dates[left-2] if left-2 >= 0 else None
            return dates[left-1], second_closest
        return dates[left-1]
    else:
        # print(f"All dates are greater for {ticker}")
        return (None, None) if fallback else None

def unitrun(dictionary, ticker, all=False, debug=False):
    if len(dictionary) == 0:
        return False,False
    if all:
        unit_list = all_units
    else:
        unit_list = valid_units
    for unit in unit_list:
        try:
            return dictionary[unit], unit 
        except KeyError:
            continue
    if all and debug:
        print(f"No unit at all available for {ticker} : {dictionary.keys()}")
    else:
        print(f"No USD like unit available for {ticker} : {dictionary.keys()}")
    if debug: 
        units = []
        for key,value in dict.items():
            units.append(key)
        units = list(set(units))
        with open(f"..\\units-checkout\\{ticker}.json", "w") as file:
            json.dump(units,file)
    return False, False

#Removes the top layer of the dict and returns the flattened version
#We are then left with only the measures, no layer on top 
def flatten(data, ticker):
    #Flatten the dataset 
    #Also removes platforms from the data and keeps staircases
    flat_data = {}
    for key, value in data["facts"].items():
        flat_data.update(value)
    # missing_count = 0
    # total_count = 0
    filtered_count = 0
    duplicate_count = 0
    for measure, datapoints_units in flat_data.items():
        datapoints, unit = unitrun(datapoints_units["units"], ticker, all=True)
        if datapoints == False:
            continue
        if len(datapoints) <3:
            continue 
        filtered = []
        duplicates = []
        end_prev = datapoints[0]["end"]
        val_prev = datapoints[0]["val"]
        # Always add the first element; comparison starts from the second element
        filtered.append(datapoints[0])
        if "start" in datapoints[0] and "start" in datapoints[-1]:
            start_prev = datapoints[0].get("start", None)
            for i in range(1, len(datapoints)):
                end = datapoints[i]["end"]
                val = datapoints[i]["val"]
                # try:
                start = datapoints[i].get("start", None)
                # except KeyError:
                #     print(measure)
                #     print(datapoints[i])
                #     print(datapoints)
                #     break
                if not (end == end_prev and val == val_prev and start == start_prev): #or datapoints[i]["form"] in ["8-K","10-K/A"]: #or end == end_next
                    filtered.append(datapoints[i]) 
                    filtered_count +=1
                else:
                    duplicates.append(datapoints[i]) 
                    duplicate_count +=1
                end_prev = end
                val_prev = val
                start_prev = start
        else:
            for i in range(1, len(datapoints)):
                end = datapoints[i]["end"]
                val = datapoints[i]["val"]
                if not (end == end_prev and val == val_prev): #or datapoints[i]["form"] in ["8-K","10-K/A"]: #or end == end_next
                    filtered.append(datapoints[i]) 
                    filtered_count +=1
                else:
                    duplicates.append(datapoints[i]) 
                    duplicate_count +=1
                end_prev = end
                val_prev = val
        flat_data[measure]["units"][unit] = filtered
        #For each company, measure, for each unfiltered datapoint if we dont have it we add one
        # for datapoint_duplicate in duplicates:
        #     gotem = False
        #     total_count +=1
        #     for datapoint_filtered in filtered:
        #         if datapoint_duplicate["val"] == datapoint_filtered["val"] and datapoint_duplicate["end"] == datapoint_filtered["end"]:
        #             gotem = True
        #     if gotem == False:
        #         # if "start" in datapoint_unfiltered.keys():
        #         missing_count +=1
    try:
        print(f"{ticker}:{(duplicate_count/(filtered_count+duplicate_count))*100}%")
    except ZeroDivisionError:
        pass
    # print(f"{ticker}:{(1-missing_count/total_count)*100}%")
    return flat_data

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

with open(r"..\other_pickle\cik.json","r") as file:
    cikdata = json.load(file)
    file.close()
    
def sync_companyfacts(ticker:str):
    cik = getcik(ticker)
    data_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data  = httpx.get(data_url, headers= headers)
    return data
    
async def companyfacts(ticker:str, client, semaphore):
    #Get all the financial data for a ticker
    cik = getcik(ticker)
    data_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = await fetch(data_url, headers, semaphore, client, TIMEOUT,RETRIES,START_RETRY_DELAY)
    return facts

async def companysubmissions(ticker:str, client, semaphore):
    #Get all the financial data for a ticker
    cik = getcik(ticker)
    data_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    facts = await fetch(data_url, headers, semaphore, client, TIMEOUT,RETRIES,START_RETRY_DELAY)
    return facts

def ratio_to_int(ratio:str):
    parts = ratio.split(":")
    current_split = int(parts[0]) / int(parts[1])
    return current_split

class Stock:
    def __init__(self, ticker:str, standard_measures):
        self.initialized_measures = {"static":[], "dynamic":[]}
        self.ticker = ticker.upper()
        self.cik = getcik(self.ticker)
        try:
            with open(os.path.join(FACTS_PATH, f"CIK{self.cik}.json"), "r") as file:
                self.data = flatten(json.load(file), self.ticker)
            with open(os.path.join(SUBMISSIONS_PATH, f"CIK{self.cik}.json"), "r") as file:
                data = json.load(file)
                self.sic = int(data["sic"])
                self.sic_desc = data["sicDescription"]
                self.foreign = False
                for form in ["20-F", "40-F", "6-K"]:
                    if form in data["filings"]["recent"]["form"]:
                        self.foreign = True
            # self.start_dates = {}
            # self.end_dates = {}
            self.measures_and_intervals = {"static":{}, "dynamic":{}}
            self.measure_paths = {"static":{}, "dynamic":{}}
            self.missing = {"static":[], "dynamic":[]}
            self.ignored = []
            self.date_dict = {"static":[], "dynamic":[]}
            self.success = self.time_init(standard_measures)
        except FileNotFoundError:
            self.success = "del"
        # except Exception as e:
        #     print(f"During Initialization: {e}")

    def time_init(self, standard_measures):
        #Also serves as a filter for the companies with wrong currencies
        #Check to see if already initialized with the measures
        overlap = {}
        extreme_start = {}
        extreme_end = {}
        needed_measures = []
        #Both static and dynamic time init
        irrelevant_measures = []
        success = {"static":0, "dynamic":0}
        flags = {"static":0, "dynamic":0}
        for motion in ["static", "dynamic"]:
            uninitialized_measures = []
            for measure in standard_measures[motion]: # Detect the measures that have not been initialized yet 
                if not measure in self.initialized_measures[motion]:
                    uninitialized_measures.append(measure)
            if uninitialized_measures == []:
                flags[motion] = 1
                continue
            # Get all the constituent measures needed through recursivefact
            for measure in uninitialized_measures:
                # print(f"Getting {measure}")
                #Make a copy.deepcopy if not working 
                if measure in deprecate_conversion:
                    measure = deprecate_conversion[measure]
                pathways = recursive_date_gather(self, measure, motion)
                if pathways["del"]:
                    return "del"
                # needed_measures += pathways["needed"]
                del pathways["del"]
                if pathways["paths"] != None:
                    self.measure_paths[motion].update({measure: pathways["paths"][0]})
                    needed_measures += pathways["paths"][2]
                    self.date_dict[motion].append(pathways["paths"][1])
                    self.measures_and_intervals[motion][measure] = pathways["paths"][1]
                else:
                    self.missing[motion].append(measure)
            if self.measure_paths[motion] == {}:
                return "del"  
            overlap[motion] = calculate_overlap(self.date_dict[motion])
            if overlap[motion] != []:
                success[motion] = 1
            else:
                success[motion] = 0
            dates = [item for sublist in self.date_dict[motion] for item in sublist]
            if dates == []:
                extreme_start[motion] = pd.Timestamp.now()
                extreme_end[motion] = pd.Timestamp.now()
            else:
                start_dates, end_dates = zip(*dates) #map(lambda x: sorted(list(x)),
                extreme_start[motion] = min(start_dates)
                extreme_end[motion] = max(end_dates)

        if flags["static"] and flags["dynamic"]: #If both static and dynamic don't have any new measures to init
            return 1,1
        #We do it akward like this because we need the to stay when flags are triggered
        self.ignored = irrelevant_measures
        self.overlap = overlap
        self.extreme_start = extreme_start
        self.extreme_end = extreme_end
        self.price_methods = []
        self.share_availability = [0,0]
        #remove duplicates
        needed_measures = list(set(needed_measures))
        #Initialize the share value as well
        for share_method, dynamic in [("EntityCommonStockSharesOutstanding",0),  ("WeightedAverageNumberOfSharesOutstandingBasic",1), ("WeightedAverageNumberOfDilutedSharesOutstanding",1)]:
            share_have = self.data.get(share_method, False)
            if share_have:
                needed_measures.append(share_method)
                self.price_methods.append(share_method)
                self.share_availability[dynamic] = 1
        self.initialized_measures = standard_measures
        #Change the strings to datetime for the initialized measures
        #Flatten batches into a single DataFrame with an identifier
        all_data = []
        for measure in needed_measures:
            if measure in deprecate_conversion:
                measure = deprecate_conversion[measure]
            datapoints, unit = unitrun(self.data[measure]["units"], self.ticker)
            if datapoints == False:
                return "del"
            for datapoint in datapoints:
                datapoint["batch_name"] = measure  # Add identifier
                all_data.append(datapoint)
        df = pd.DataFrame(all_data)
        df['filed'] = pd.to_datetime(df['filed'], format='%Y-%m-%d')
        df['end'] = pd.to_datetime(df['end'], format='%Y-%m-%d')
        df['start'] = pd.to_datetime(df['start'], format='%Y-%m-%d', errors='coerce')
        separated_batches = {name: df[df['batch_name'] == name].drop(columns=['batch_name']).to_dict('records') for name in needed_measures}
        self.converted_data = separated_batches
        return (success["static"], success["dynamic"])
    
    def date_reset(self):
        self.initialized_measures = {"static":[], "dynamic":[]}

    async def async_init(self,client, semaphore, standard_measures):
        #Get all of the data for the company, ALL of it 
        data = await companysubmissions(self.ticker, client, semaphore)
        #If the response wasn't recieved, skips the rest of the code 
        if type(data) == str:
            return "del"
        elif type(data) != int:
            data = data.json()
            self.sic = data["sic"]
            self.sicDescription = data["sicDescription"]
            return 1
    
    async def price_init(self,semaphore):
        #Get the price and set the self.price
        try:
            self.fullprice = await yahoo_fetch(self.ticker, self.extreme_start, self.extreme_end, semaphore, RETRIES, START_RETRY_DELAY)
            if type(self.fullprice) == int:
                return 0
            self.info = await yahoo_info_fetch(self.ticker, RETRIES, START_RETRY_DELAY)
            if type(self.info) == int:
                return 0
            splits = await yahoo_split_fetch(self.ticker, RETRIES, START_RETRY_DELAY)
            if type(splits) == int:
                return 0
            self.dividends = await yahoo_dividend_fetch(self.ticker, RETRIES, START_RETRY_DELAY)
            if type(self.dividends) == int:
                return 0
            if not splits.empty:
                splits["splitRatio"] = splits["splitRatio"].apply(ratio_to_int)
            self.splits = splits
            Price = self.fullprice[["close", "adjclose"]].copy()
            self.price = Price.ffill().bfill()
            return 1 
        except asyncio.TimeoutError as ex:
            print(f"Timeout: {ex}")
            return 0
        except asyncio.CancelledError as ex:
            print(f"Canceled: {ex}")
            return 0
        except Exception as e:
            print(f"price init {e}")
            return 0
    
    def fact(self, measure, intervals = None, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5, annual=False, reshape_approx=False, date_gather=False, forced_static=0, share_filter=0, forced_dynamic=0):
        """  
        If date_gather, then it returns a dataframe to allow recursive_fact to get the date.
        Returns a dataframe that has rows indexed row_delta away, with lookbehind columns that are column_delta away.
        If the data is dynamic then the row and column deltas are fixed.
        Dynamic tolerance is how much into the future the price we are predicting is.
        If intervals is None, that means get as much as possible
        """
        #Propagate the 0 
        if self.data == 0:
            return 0
        # try:
        #If the measure is deprecated switch to the undeprecated version
        if measure in deprecate_conversion:
            measure = deprecate_conversion[measure]
            # frame = pd.concat([frame, frame_undep], axis=0).reset_index(drop=True)
        if date_gather:
            if measure in self.data:
                data= self.data[measure]
            elif measure in irrelevants[get_category(self.sic)]:
                return "ignored"
            else:
                return None
            point_list, unit = unitrun(data["units"], self.ticker)
            if point_list == False:
                # print(list(data["units"].keys()))
                return "del"
            annual_flag = False
            for annual_measure in annual_measures:
                if measure in measure_conversion.get(annual_measure,[]) or measure in approximate_measure_conversion.get(annual_measure,[]):
                    annual_flag = True
                    break
            if measure in annual_measures or annual_flag:
                approx = True
            else:
                approx = False
            reshaped, intervals, dynamic = reshape(measure, point_list, self.ticker, converted=False, approx=approx, forced_dynamic=forced_dynamic)
            if intervals == []:
                return None
            return intervals
        if measure in self.converted_data:
            data= self.converted_data[measure]
            converted= True
        elif measure in irrelevants[get_category(self.sic)]:
            base_dates = pd.date_range(start=self.extreme_start["static"], end=self.extreme_end["static"], freq=row_delta)
            frame = pd.DataFrame(index =base_dates, columns = [f"{measure}-{i}" for i in range(0,lookbehind)])
            frame = frame.infer_objects()
            frame.fillna(0,inplace = True)
            return frame, "ignored"
        elif measure in self.data:
            data = unitrun(self.data[measure]["units"], self.ticker)
            converted = False
        else:
            # print(f"{self.ticker}: Data not converted or available for {measure}")
            return pd.DataFrame(), None
        #Get the index dates for the datpoints for measure
        #If the measure doesn't make sense for the company's category, replace all of the datapoints with 0
        annual_flag = False
        for annual_measure in annual_measures:
            if measure in measure_conversion.get(annual_measure,[]) or measure in approximate_measure_conversion.get(annual_measure,[]):
                annual_flag = True
                break
        if measure in annual_measures or annual_flag:
            reshape_approx = True        
        reshaped, all_intervals, dynamic = reshape(measure, data, self.ticker, annual, reshape_approx, converted=converted, forced_static=forced_static, forced_dynamic=forced_dynamic) # _ is intervals, if the data has not been converted in the init, it will be here
        if intervals == None:
            intervals = all_intervals 
        if intervals == []:
            print(f"intervals empty for {measure}")
        if forced_static:
            dynamic = False
            #We also have to swap the start and end because we want to agknowledge the data from the start
            reshaped = {v[-1]['start']: [{**v[-1], 'end': k}] for k, v in reshaped.items()} 
        if share_filter:
            #If there are on splits this is just an empty frame and everything defaults to 1
            splits = self.splits
            #If we are getting the share data and we recieved the splits frame
            #The split is always between the first date and onward 
            current_shares = self.info.get('sharesOutstanding') 
            reshaped[pd.Timestamp.now()] = [{"val": current_shares, "start": pd.Timestamp.now(), "filed": pd.Timestamp.now()}] #This is our anchor to reality, we know this needs to be treu and then we can check the rest relative to this
            iterator = list(reshaped.items()) 
            insertions = [] #We need to accelerate the split faster then when its reported by inserting before it
            errors = []
            index = len(iterator) - 1  # Start from the end of the list
            gap = 1
            #Prev things are the things previously in time but ahead in the iteration
            while index - gap >= 0:  # Ensure index is within the bounds of the list
                date, point_list = iterator[index]
                pos = splits.index.searchsorted(date, side='right') - 1  # Find the closest split before or at date
                value = point_list[-1]["val"]
                prev_date, prev_point_list = iterator[index - gap]
                prev_value = prev_point_list[-1]["val"]
                if pos >= 0 and prev_date < splits.index[pos]:  # Check if there is a split between the date and prev_date
                    current_split = splits.iloc[pos]["splitRatio"]
                    insertions.append({splits.index[pos]:[{"val":value, "filed": pd.Timestamp.now()}]}) #We just extend the value that we already have backwards to the split date
                else:
                    current_split = 1  # No split affects this period
                if prev_value < 1:
                    errors.append(prev_date)
                    gap += 1  
                    continue
                ratio = value / prev_value
                # if not forced_static:
                #     print(current_split, ratio)
                if current_split * (0.80 - gap*0.05) < ratio < current_split * (1.20 + gap*0.05):  # Check if the ratio is withing bounds of the split that is expected, it increases with the gap that has happened
                    index -= gap  # Move to the next item (backwards)
                    gap = 1
                else:
                    errors.append(prev_date)
                    gap += 1  # Increase gap to skip further back

            for error in errors:
                del reshaped[error]
            # for insertion in insertions:
            #     reshaped.update(insertion)
            # reshaped = dict(sorted(reshaped.items())) #Maintain order

            #Here we multiply back to get the shares adjusted for splits 
            iterator = list(reshaped.items())
            for date, list_of_dict in iterator:
                pos = splits.index.searchsorted(date, side='right')
                if pos > 0:
                    split_multiple = splits.iloc[pos:]["splitRatio"].prod()
                else:
                    split_multiple = 1
                for i, datapoint in enumerate(list_of_dict):
                    datapoint["val"] = datapoint["val"] * split_multiple
                    reshaped[date][i] = datapoint

        for k ,interval in enumerate(intervals): #To save compute we only get the data that we really need, could be intervals with gaps
            data_start, data_end = interval
            if dynamic:
                row_delta = dynamic_row_delta
                column_delta = pd.Timedelta(days=95)
            tolerance = dynamic_tolerance * int(dynamic) + static_tolerance * (int(not dynamic))
            data_start = data_start - lookbehind * column_delta - pd.Timedelta(days=5) #To account for the lookbehind, it does not matter if it is way too much, it will just turn out blank and we can handle it later
            if not dynamic and not forced_dynamic: #static 
                dates = list(reshaped.keys())
                base_dates = pd.date_range(start=data_start, end=data_end, freq=row_delta) #No longer Inverted start and end to keep the correct order
                if base_dates.empty:
                    print("The interval doesn't fit with frequency")
                i_values = np.arange(lookbehind)  # An array [0, 1, ..., lookbehind-1]
                adjustments = i_values * column_delta
                # Broadcasted subtraction: for each date in base_dates, subtract each value in adjustments
                row_dates = np.subtract.outer(base_dates, adjustments)
                dimensions = row_dates.shape
                flat_row_dates = row_dates.flatten()
                date_series = pd.Series(flat_row_dates)
                # Use apply to run closest_date on each date
                results_series = date_series.apply(lambda x: closest_date(tuple(dates), x, self.ticker, fallback=True))  # m rows and n columns, replace with actual values
                row_indexes = results_series.to_numpy().reshape(dimensions)
                row_indexes = pd.DataFrame(row_indexes, index =base_dates)
                #each item is a tuple of dates 
                frame_dict = {row_index:[] for row_index in base_dates}
            else:
                timestamps = list(reshaped.keys()) 
                timestamps = [timestamp for timestamp in timestamps if (data_start <= timestamp <=data_end)] #filter for the interval
                index_length = len(timestamps) 
                unpadded_timestamps = timestamps
                timestamps = [None] * lookbehind + timestamps #Padding
                rows = [zip(timestamps[lookbehind:0:-1], [None]*lookbehind)] #The first few rows
                for i in range(1, index_length):
                    window = zip(timestamps[i+lookbehind:i:-1], [None]*lookbehind)
                    rows.append(window)
                row_indexes = pd.DataFrame(rows, index = unpadded_timestamps)
                frame_dict = {row_index:[] for row_index in unpadded_timestamps}
                
            for row_index, row in row_indexes.iterrows():
                barrier = row_index + tolerance
                for index_tuple in row: 
                    index, fallback_index = index_tuple
                    nearest_filed = pd.Timestamp.min
                    uptodate = {"val": np.nan}
                    if index!= None:
                        for value in reshaped[index]:  
                            if nearest_filed < value["filed"] <= barrier:
                                uptodate = value
                                nearest_filed = value["filed"]
                    if np.isnan(uptodate["val"]):
                        if fallback_index != None:
                            for value in reshaped[fallback_index]:  
                                if nearest_filed < value["filed"] <= barrier:
                                    uptodate = value
                                    nearest_filed = value["filed"]
                    frame_dict[row_index].append(uptodate["val"])
            frame = pd.DataFrame.from_dict(frame_dict, orient='index')
            if frame.empty:
                print(f"{measure} inelligible for {self.ticker}")
                total_frame = pd.DataFrame(columns=[i for i in range(lookbehind)])
            elif k == 0:
                total_frame = frame
            elif k != 0:
                total_frame = total_frame.combine_first(frame)
        total_frame.columns = [f"{measure}-{i}" for i in range(0,lookbehind)]
        return total_frame, "Some_unit"   
        # except KeyError as e:
        #     print(f"in fact keyerror: {e}")
        #     print(f"Fact {measure} not available for {self.ticker}.")
        # except Exception as e:
        #     print(f"in fact: {e}")

def frame_rename(frame,name):
    renamed_columns = [] 
    for column in frame.columns:
        parts = column.split("-", 1)  # Splits at the first "-"
        parts[0] = name  # Replace the first part
        renamed = "-".join(parts)
        renamed_columns.append(renamed)
    frame.columns = renamed_columns
    return frame

def data_concat(frames_intervals_list):
    for frame, interval in frames_intervals_list:
        pass

def operate(comp, frames, frame_names, operation, measure, dynamic, approx, approx_needed = 0):
    #Rename the frames so we can easily use them and check if they are empty to prevent errors
    frames_and_names = [(frame_rename(frame, measure),frame_name) for frame, frame_name in zip(frames,frame_names) if frame.empty == False]
    frames, frame_names = zip(*frames_and_names)
    if operation == "sub" and len(frames) != 2:
        print("Need two frames to sub, the empty check could have triggered")
    if dynamic == False: #This alignment doesn't work with static data that is a few days apart because we filter the days that are close to each other
        values = list(frames)
        if not approx:
            if operation == "add":
                #Get all the possible indexes for the frame
                frame1 = values[0]
                for idx, value in enumerate(values[1:], start=1):
                    frame1, value = frame1.align(value, join="outer", axis=0)
                    values[idx] = value
                result_frame = pd.DataFrame(index=frame1.index)
                for i, col in enumerate(frame1.columns):
                    result_frame[f'{measure}-{i}'] = frame1[col]  # Initialize with frame1's columns {i - lookbehind +1}
                for value in values[1:]:
                    for i, col in enumerate(value.columns):
                        result_frame[f'{measure}-{i}'] = result_frame[f'{measure}-{i}'].add(value[col])
                return result_frame
            
            if operation == "sub":
                add, sub = values[0], values[1]
                add, sub = add.align(sub, join="outer", axis=0)
                result_frame = pd.DataFrame(index=add.index)
                for i, col in enumerate(add.columns):
                    result_frame[f'{measure}-{i}'] = add[col]  # Initialize with add's columns
                for i, col in enumerate(sub.columns):
                    result_frame[f'{measure}-{i}'] = result_frame[f'{measure}-{i}'].sub(sub[col])
                return result_frame
        else:
            if operation == "add":
                #Get all the possible indexes for the frame
                frame1 = values[0]
                for idx, value in enumerate(values[1:], start=1):
                    frame1, value = frame1.align(value, join="outer", axis=0)
                    values[idx] = value
                result_frame = pd.DataFrame(index=frame1.index)
                for i, col in enumerate(frame1.columns):
                    result_frame[f'{measure}-{i}'] = frame1[col]  # Initialize with frame1's columns {i - lookbehind +1}
                result_frame["checker"] = 1
                for value in values[1:]:
                    value["checker"] = value.apply(lambda row: 0 if row.isnull().any() else 1, axis=1) #You can implement fractions
                    value.fillna(0, inplace = True)
                    for i, col in enumerate(value.columns[:-1]):
                        result_frame[f'{measure}-{i}'] = result_frame[f'{measure}-{i}'].add(value[col], fill_value = 0) #fill_value Specific for approx 
                    result_frame["checker"] = result_frame["checker"].add(value["checker"], fill_value = 0)
                columns_to_replace = result_frame.columns[:-1]  
                final_frame = pd.DataFrame(index=result_frame.index)
                for col in columns_to_replace:
                    final_frame[col] = np.where(result_frame["checker"] >= approx_needed, result_frame[col], np.nan) #We want to keep the gaps in the frame for addition and stuff
                return final_frame
    #First we find the intervals for the data that we have 
    intervals_list = []
    # for name in frame_names:
        # intervals = comp.measures_and_intervals["dynamic"][name] #We are in dynamic anyway
        # intervals_list.append(intervals)
    for frame in frames:
        # Calculate the difference in days directly from the index
        date_diff = frame.index.to_series().diff().dt.days
        
        # Identify new intervals based on the 100-day threshold
        new_interval = date_diff > 100
        # Identify start dates for intervals directly from the index
        first_valid_index = frame.apply(pd.Series.first_valid_index).min()
        start_dates = frame.index[new_interval | (frame.index == first_valid_index)]
        
        # Calculate end dates by shifting start dates directly in the index
        new_interval_indices = new_interval[new_interval | (frame.index == first_valid_index)].index
    
        # We ignore the first start and then add the last starts end at the end
        end_dates = []
        for idx in new_interval_indices[1:]: 
            # Find the position of the current index and access the previous index unless it's the first index
            pos = frame.index.get_loc(idx)
            end_dates.append(frame.index[pos - 1])
        if frame.index[-1] not in end_dates:
            end_dates.append(frame.index[-1])
        # Combine into a list of tuples
        intervals = list(zip(start_dates, end_dates))
        intervals_list.append(intervals)

    frames_and_intervals = list(zip(frames,intervals_list))
    if not approx: 
        # calc the real overlap intervals then iterate through them and add the stuff back,
        extended_intervals_list = []  #We extened the intervals since the start and end could be a few days apart
        for intervals in intervals_list:
            extended_intervals = []
            for interval in intervals:
                start, end = interval
                extended_intervals.append((start - pd.Timedelta(days=5), end + pd.Timedelta(days=5)))
            extended_intervals_list.append(extended_intervals)
        overlap_intervals = calculate_overlap(extended_intervals_list)
        total_frame = pd.DataFrame()
        for overlap in overlap_intervals:
            overlap_start, overlap_end = overlap
            # This makes sure that the real overlap is included
            # extended_start = overlap_start - pd.Timedelta(days=5)
            # extended_end = overlap_end + pd.Timedelta(days=5)
            extended_start = overlap_start
            extended_end = overlap_end
            temp_frames = [frame[extended_start:extended_end] for frame in frames]
            reasign_index = temp_frames[0].index 
            temp_frames = [frame.reset_index(drop=True) for frame in temp_frames]
            if operation == "sub":
                resulting_frame = temp_frames[0] - temp_frames[1]
            if operation == "add":
                resulting_frame = temp_frames[0]
                for frame in temp_frames[1:]:
                    resulting_frame += frame
            resulting_frame.index = reasign_index
            total_frame = total_frame.combine_first(resulting_frame)
        return total_frame
    
    else: #We keep residuals here 
        #Make a total index that we will use as the canvas for all the operations
        all_indices = pd.concat([pd.Series(frame.index) for frame in frames if not frame.index.empty])
        index_combined = pd.Index(all_indices.dropna().unique())
        total_index = index_combined.sort_values()
        differences = total_index.to_series().diff()
        differences.iloc[0] = pd.Timedelta(days=91) #To keep the first value 
        filtered_index = total_index[differences > pd.Timedelta(days=5)] #We do this so we they are all spaced enough apart 
        starting_frame = pd.DataFrame(0, index=filtered_index, columns =frames[0].columns) #Make it the same as previous frames
        starting_frame = starting_frame
        starting_frame["checker"] = 0
        for frame in frames:
            frame["checker"] = 1
            frame["checker"] = (~frame.isnull().any(axis=1)).astype(int) #You can implement fractions
            frame.fillna(0, inplace = True)
        if operation == "add":
            for frame, intervals in frames_and_intervals:
                for interval in intervals:
                    start, end = interval
                    extended_start = start - pd.Timedelta(days=5)
                    extended_end = end + pd.Timedelta(days=5)
                    reasign_index = starting_frame[extended_start:extended_end].index
                    temp_frame = frame[extended_start:extended_end]
                    temp_frame.index = reasign_index
                    starting_frame = starting_frame.add(temp_frame, fill_value=0)
            columns_to_replace = starting_frame.columns 
            final_frame = pd.DataFrame(index =starting_frame.index)
            for col in columns_to_replace:
                final_frame[col] = np.where(starting_frame["checker"] >= approx_needed, starting_frame[col], np.nan) #We want to keep the gaps in the frame for addition and stuff
            # final_frame = starting_frame[starting_frame["checker"] >= approx_needed] 
            final_frame.drop(columns=["checker"], inplace=True)
            return final_frame
        else:
            print("Only add is possible with approx")

def path_selector(comp, measure, path, dynamic, intervals = None, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5 , annual=False, reshape_approx = False, forced_dynamic=False):
    """
    Takes in the desired path to the data and outputs the data
    New recursive_fact
    There is no approx since the approximity of the path is determined by recursive_date_gather
    """
    #[({'Assets': True}, ('2008-09-27', '2023-12-30')), ({'LiabilitiesAndStockholdersEquity': True}, ('2008-09-27', '2023-12-30')), ({'add': {'AssetsCurrent': True, 'AssetsNoncurrent': {'sub': {'Assets': True, 'AssetsCurrent': True}}}}, ('2008-09-27', '2023-12-30'))]
    #[{'Assets': (True, ('2008-09-27', '2023-12-30'))]}, ({'LiabilitiesAndStockholdersEquity': (True, ('2008-09-27', '2023-12-30'))}), ({'add': {'AssetsCurrent': (True,('2008-09-27', '2023-12-30')), 'AssetsNoncurrent': {'sub': {'Assets': True, 'AssetsCurrent': True}}}})]
    # if intervals == None:
    #     intervals = (pd.Timestamp(year=1970, month=1, day=1), pd.Timestamp.now())
    for root, tree in path.items():
        if tree == True:            
            value, unit = comp.fact(root, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
            value = frame_rename(value,measure)
            return value, unit
        #Concat path
        elif root == "concat":
            fledgling, get_interval = tree[0]
            if intervals != None:
                new_intervals = calculate_overlap([intervals, get_interval])
            else:
                new_intervals = get_interval
            if new_intervals != []:
                concat_frame, unit = path_selector(comp, measure, {measure:fledgling}, dynamic, new_intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
            else:
                concat_frame = pd.DataFrame() #We need something to add so we have a placeholder
            for sapling in tree[1:]:
                fledgling, get_interval = sapling
                if intervals != None:
                    new_intervals = calculate_overlap([intervals, get_interval])
                    if new_intervals == []:
                        continue
                else:
                    new_intervals = get_interval
                frame, unit = path_selector(comp, measure, {measure:fledgling}, dynamic, new_intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
                if frame.empty:
                    continue
                if not concat_frame.empty:
                    frame.columns = concat_frame.columns
                    concat_frame = concat_frame.combine_first(frame)
                else:
                    concat_frame = frame
            if dynamic or forced_dynamic: #Filter the dynamic data after concatting, there could be datpoinst days apart
                filtered_dates = []
                last_date = None
                for index, row in concat_frame.iterrows():
                    if last_date is None or (index - last_date).days >= 85:
                        filtered_dates.append(row)
                        last_date = index
                        last_score = row.notna().sum()
                    else:
                        score = row.notna().sum()
                        if score > last_score:
                            filtered_dates.pop()
                            filtered_dates.append(row)
                concat_frame = pd.DataFrame(filtered_dates)
            concat_frame = frame_rename(concat_frame, measure)
            return concat_frame, unit  #The units are the same...
        #Add path
        elif root == "add":
            values = []
            frame_names = []
            for sprout, tree_part in tree.items():
                add_value, unit = path_selector(comp, sprout, {sprout:tree_part}, dynamic, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
                values.append(add_value)
                frame_names.append(sprout)
            result_frame = operate(comp, values, frame_names, "add", measure, dynamic or forced_dynamic, False)
            result_frame = frame_rename(result_frame,measure)
            return result_frame, unit
            
        elif root == "sub":
            larch = list(tree.items())
            add, unit = path_selector(comp, larch[0][0], {larch[0][0]:larch[0][1]}, dynamic, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
            sub, unit = path_selector(comp, larch[1][0], {larch[1][0]:larch[1][1]}, dynamic, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
            #Get all the possible indexes for the frame
            values = [add, sub]
            frame_names = [larch[0][0], larch[1][0]]
            result_frame = operate(comp, values, frame_names, "sub", measure, dynamic or forced_dynamic, False)
            result_frame = frame_rename(result_frame,measure)
            return result_frame,unit
        
        elif root == "approx_add":
            values = []
            frame_names = []
            for sprout, tree_part in tree.items():
                add_value, unit = path_selector(comp, sprout ,{sprout:tree_part}, dynamic, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
                #Specifically for approx
                # add_value.fillna(0, inplace =True)
                values.append(add_value)
                frame_names.append(sprout)
            threshold = approximate_additive_conversion[measure][1] #Get the wanted threshold for the frames and shit
            result_frame = operate(comp, values, frame_names, "add", measure, dynamic or forced_dynamic, True, threshold)
            result_frame = frame_rename(result_frame,measure)
            #Specifically for approx
            result_frame.replace(0,np.nan, inplace=True)
            return result_frame, unit

        #serves as the measure conversion part
        else:
            return path_selector(comp, root, tree, dynamic, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)

# Function to find overlap between two intervals
def find_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    if start1 <= end2 and start2 <= end1:  # Check if there is an overlap
        return (max(start1, start2), min(end1, end2))
    return None

# Function to intersect current common intervals with a new list of intervals
def intersect_with_common(current_common, new_intervals):
    temp_common = []
    for common_interval in current_common:
        for new_interval in new_intervals:
            overlap = find_overlap(common_interval, new_interval)
            if overlap:
                temp_common.append(overlap)
    return temp_common

def calculate_overlap(list_of_interval_lists):
    """Calculate the total overlap across multiple lists of intervals."""
    if not list_of_interval_lists:
        return []
    # Start with the intervals in the first list
    common_intervals = list_of_interval_lists[0]
    # Iterate through the remaining lists
    for intervals in list_of_interval_lists[1:]:
        common_intervals = intersect_with_common(common_intervals, intervals)
        if not common_intervals:  # No overlap left, early exit
            break
    return common_intervals
  
def path_finder(paths, constituent=False):
    if paths == None:
        return None
    if constituent:
        intervals = [constit[1] for constit in paths]
        overlap = calculate_overlap(intervals)
        if overlap == []:
            return None
        needed_list = []
        for constit in paths:
            needed_list += constit[2]
        return [part[0] for part in paths], overlap, list(set(needed_list))
    
    else:   
        if not len(paths): #If the result is empty
            return None
        if len(paths) == 1:
            return paths[0]
        components = []
        for path in paths:
            path, intervals, needed = path
            for interval in intervals:
                components.append((path, interval, needed))
        components = list(filter(lambda x: x[1][0]<x[1][1], components)) #Filters the intervals where end is before start
        if not len(components):
            return None
        #We combine them to a concat list and pass that back
        #We have to sort them by start, then append them one by one 
        sorted_paths = sorted(components, key=lambda x: x[1][0])
        concat_paths = []
        needed_measures = []
        intervals = []
        first_index = 0
        #First we find the best starting candidate 
        first_path, (first_start, first_end), first_needed = sorted_paths[0]
        for i, route in enumerate(sorted_paths[1:], start=1):
            path, (start, end), needed = route
            if start == first_start:    
                if start <= first_end and end > first_end or (end == first_end and (len(needed)<len(first_needed))): #or (len(needed)==len(best[2]) and analyze_structure(route) < analyze_structure(best)
                    first_path = path
                    first_start = start
                    first_end = end
                    first_needed = needed
                    first_index = i
        concat_paths.append((first_path,[(first_start,first_end)]))
        needed_measures += first_needed
        total_start = first_start
        current_end = first_end
        idx = first_index
        appended_flag = False # This is set to true when the previous itteration has already added the interval to intervals
        for _ in range(first_index +1, len(sorted_paths)):
            if idx == len(sorted_paths) -1:
                break
            best = None 
            best_start = None
            best_end = current_end
            #Find the next best candidate
            for k, route in enumerate(sorted_paths[idx+1:], start=idx+1):
                path, (start, end), needed = route
                #When there is no candidate we search for one and only when there is an eligible candidate do we try to replace it
                if not best:
                    if (start <= current_end + pd.Timedelta(days=4)) and end > current_end: #Cut some slack for gaps worth a day
                        idx = k
                        best = route
                        best_start = start
                        best_end = end
                        best_needed = needed
                    else:
                        continue
                elif (start <= current_end + pd.Timedelta(days=4)):
                    if end > best_end:
                        idx = k
                        best = route
                        best_start = start
                        best_end = end
                        best_needed = needed
                        # if route in concat_paths: #Prevent further tests and just add it in there, because we have already used it, the check means that we used an interval through the same method
                        #     break
                    elif end == best_end and (len(needed)<len(best_needed)):
                        idx = k
                        best = route
                        best_start = start
                        best_end = end
                        best_needed = needed
            if best == None: #since that means there are no other eligible intervals to add 
                if appended_flag == False:
                    intervals.append((total_start, current_end)) #Add the closed interval we got 
                first_path, (first_start, first_end), first_needed = sorted_paths[idx+1]
                first_index = idx+1 
                for i, route in enumerate(sorted_paths[idx+2:], start=idx+2):
                    path, (start, end), needed = route
                    if start == first_start:    
                        if (start <= first_end + pd.Timedelta(days=4)) and end > first_end or (end == first_end and (len(needed)<len(first_needed))): #or (len(needed)==len(best[2]) and analyze_structure(route) < analyze_structure(best)
                            first_path = path
                            first_start = start
                            first_end = end
                            first_needed = needed
                            first_index = i
                if first_end > current_end:
                    concat_paths.append((first_path,[(first_start,first_end)]))
                    needed_measures += first_needed
                    total_start = first_start
                    current_end = first_end
                    if first_index == len(sorted_paths) -1:
                        intervals.append((total_start, current_end))
                        break
                    else:
                        appended_flag = False
                else:
                    idx += 1 #Go to the next iteration
                    appended_flag = True
            else: 
                current_end = best_end
                concat_paths.append((best[0], [(best_start,best_end)]))
                if idx == len(sorted_paths) -1:
                    intervals.append((total_start, best_end))
                needed_measures += best_needed
        if intervals == []: #If we also used the last interval we still have to append it 
            intervals.append((total_start, current_end))

        # Sort tuples by the string representation of the identifier
        concat_paths.sort(key=lambda x: str(x[0]))

        # Assuming you want to just gather lists under the same key
        combined_tuples = []
        current_key = None
        current_list = []

        for key, lst in concat_paths:
            if key != current_key:
                if current_key is not None:
                    combined_tuples.append((current_key, current_list))
                current_key = key
                current_list = lst[:]
            else:
                current_list.extend(lst)

        # Don't forget to add the last group if it exists
        if current_list:
            combined_tuples.append((current_key, current_list))

        return ({"concat":combined_tuples}, intervals, list(set(needed_measures)))
    
#list({json.dumps(d, sort_keys=True): d for d in concat_paths}.values())
def analyze_structure(structure, depth=1):
    """
    Recursively analyze a nested structure (dicts, lists, tuples, sets) to find:
    - Maximum depth of nesting
    - Total number of elements
    """
    max_depth = depth
    total_elements = 0
    # Determine the type of the structure and iterate accordingly
    if isinstance(structure, (dict,)):
        iterable = structure.items()
    elif isinstance(structure, (list, tuple, set)):
        iterable = enumerate(structure)
    else:
        return depth, 1
    for _, value in iterable:
        if isinstance(value, (dict, list, tuple, set)):
            # If the value is also a container, recurse
            current_depth, current_elements = analyze_structure(value, depth + 1)
            max_depth = max(max_depth, current_depth)
            total_elements += current_elements
        else:
            total_elements += 1
    if depth ==1:
        return max_depth + total_elements
    return max_depth, total_elements

def calculate_total_duration(intervals):
    total_duration = pd.Timedelta(days=0)  # Initialize total_duration to zero
    for start, end in intervals:
        if end > start:  # Ensure the end time is after the start time
            duration = end - start
            total_duration += duration  # Add the duration of this interval to the total
    return total_duration

#Used: {"conversion":["bla", "blabla"], "addition":[blabla]}
def recursive_date_gather(comp, measure, motion, depth=0, path_date=None, approx = True, printout =False):
    #path = ("first_step", "second_step"...)
    #The whole thing returns the best path with the date that you get
    #Individual calls return the best dates and the measures needed to get it as a tuple
    # dates = (dates, needed)
    #resulting paths look like ({replacement:{sub:{first:None, second:{replacement:{}}}}, (start,end))
    #True means the path ends there
    #You do not pass the path down you pass it up
    #We create the paths at the top and not at the bottom
    #The needed measures will be included in the whole path
    paths = []
    if depth ==0:
        if path_date is None:
            path_date = {"del": False, "paths":[], "ignored":[]}
    if depth>4:
        return None
    if printout:
        print(f"Entering recursive with {measure}")
    dates = comp.fact(measure, date_gather = True)
    if dates == "del":
        path_date["del"] = True
        if depth==0:
            return path_date
        return None
    if dates == "ignored":
        if depth==0:
            paths.append(({measure:True},[(pd.Timestamp(year=1970, month=1, day=1), pd.Timestamp.now())],[])) #Just to make path seeking better
        else:   
            paths.append((True,[(pd.Timestamp(year=1970, month=1, day=1), pd.Timestamp.now())],[]))
        path_date["ignored"].append(measure)
    elif dates != None:
        if depth==0:
            paths.append(({measure:True},dates,[measure])) #Just to make path seeking better
        else:   
            paths.append((True,dates,[measure]))
        
    if measure in measure_conversion:
        for replacement in measure_conversion[measure]:
            path = recursive_date_gather(comp, replacement, motion, depth+1, path_date, approx, printout)
            # path = path_finder(path)
            if path != None:
                paths.append(({replacement:path[0]},path[1],path[2]))

    if measure in additive_conversion:
        branches = []
        parts = []
        if additive_conversion[measure] != []:
            abort = False
            for part in additive_conversion[measure]:
                if abort == False:
                    path = recursive_date_gather(comp, part, motion, depth+1,path_date, approx, printout) 
                    if path == None:
                        if part not in optional:
                            abort = True
                        continue
                    else:
                        branches.append(path)
                        parts.append(part)
            if not abort:
                stuff = path_finder(branches,True)
                if stuff:
                    path, interval, needed = stuff
                    if depth == 0:
                        paths.append(({measure:{"add":{part:path[i] for i,part in enumerate(parts)}}}, interval, needed))
                    else:
                        paths.append(({"add":{part:path[i] for i,part in enumerate(parts)}}, interval, needed))

    if measure in subtract_conversion:
        path_add = recursive_date_gather(comp,subtract_conversion[measure][0], motion, depth+1, path_date, approx, printout)
        path_sub = recursive_date_gather(comp,subtract_conversion[measure][1], motion, depth+1, path_date, approx, printout)
        if path_add != None and path_sub != None:
            stuff = path_finder([path_add, path_sub], True)
            if stuff:
                (path_add,path_sub), interval, needed = stuff
                if depth == 0:
                    paths.append(({measure:{"sub":{subtract_conversion[measure][0]:path_add,subtract_conversion[measure][1]:path_sub}}},interval, needed))
                else:
                    paths.append(({"sub":{subtract_conversion[measure][0]:path_add,subtract_conversion[measure][1]:path_sub}},interval, needed))
    if approx:
        if measure in approximate_measure_conversion:
            for replacement in approximate_measure_conversion[measure]:
                path = recursive_date_gather(comp, replacement, motion, depth+1, path_date, approx, printout)
                # path = path_finder(path)
                if path != None:
                    paths.append(({replacement:path[0]},path[1],path[2]))     
        if measure in approximate_additive_conversion:
            branches = []
            parts = []  #We need to record which parts we actuall used
            optionals = 0 #To calc the percentage of optionals used
            approxes, parts_needed = approximate_additive_conversion[measure]
            if approxes != []:
                abort = False
                for part in approxes:
                    if abort == False:
                        path = recursive_date_gather(comp, part, motion, depth+1,path_date, approx, printout) 
                        if path == None: #Check if start is before end path[1][0] > path[1][1]
                            if part not in optional:
                                abort = True
                            else:
                                optionals+=1
                            continue
                        else:
                            branches.append(path)
                            parts.append(part)
            if not abort and (len(approxes)-optionals) >= parts_needed: #To prevent ridiculous approxes 
                stuff = path_finder(branches,True)
                if stuff:
                    path, interval, needed = stuff
                    if depth == 0:
                        paths.append(({measure:{"approx_add":{part:path[i] for i,part in enumerate(parts)}}}, interval, needed))
                    else:
                        paths.append(({"approx_add":{part:path[i] for i,part in enumerate(parts)}}, interval, needed))            
    if depth ==0:
        if path_date["del"] == True:
            return path_date
        found_paths = path_finder(paths)
        path_date["paths"] = found_paths
        return path_date    
    
    if paths == []:
        return None
    found_paths = path_finder(paths)
    if found_paths != None:
        previous = comp.measures_and_intervals[motion].get(measure, False)
        if not previous:
            comp.measures_and_intervals[motion][measure] = found_paths[1]
        elif calculate_total_duration(previous) < calculate_total_duration(found_paths[1]):
            comp.measures_and_intervals[motion][measure] = found_paths[1]
    return found_paths

def get_category(sic):
    for category, number_ranges in categories.items():
        for number_range in number_ranges:
            if number_range[0]<=sic<=number_range[-1]:
                return category
    return "Uncategorized"

months_dict = {1: 90, 2: 92, 3: 91, 4: 92, 5: 92, 6: 92, 7: 92, 8: 91, 9: 92, 10: 92, 11: 90, 12: 90}
#(comp, measure, depth=0, approx = True, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5 , annual=False, printout =False, date_gather= False
def acquire_frame(comp, measures:dict, available, indicator_frame, reshape_approx= True, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365), static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91),  lookbehind =5, annual=False, forced_dynamic=False):
    #Get a dataframe from the saved data of some stock 
    #Returns 0 in all the columns where data is missing
    comp.time_init(measures)
    catg = get_category(comp.sic)
    frames_dict = {"static":[], "dynamic":[]}
    unit_dict = {"static":[], "dynamic":[]}
    df = {"static": pd.DataFrame(), "dynamic": pd.DataFrame()}
    motions = ["static", "dynamic"]
    motions = [motions[i] for i, condition in enumerate(available) if condition != 0]
    for motion in motions:
        if motion == "dynamic":
            dynamic = True
        else:
            dynamic = False
        for measure in measures[motion]:
            if measure not in comp.missing[motion]:
                print(f"{comp.ticker} Getting {measure}")
                print(comp.measure_paths[motion][measure])
                # intervals = comp.measures_and_intervals[motion][measure]
                if motion == "static": #To account for forced dynamic
                    data, unit = path_selector(comp, measure, comp.measure_paths[motion][measure], dynamic, None, row_delta , column_delta , static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=forced_dynamic)
                else:
                    data, unit = path_selector(comp, measure, comp.measure_paths[motion][measure], dynamic, None, row_delta , column_delta , static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx, forced_dynamic=False)
                data.name = measure
                frames_dict[motion].append(data)
                unit_dict[motion].append(unit)
        if frames_dict[motion] == []:
            break
        print(f"Concatting {motion}")
        if motion == "static" and not forced_dynamic:
            df[motion] = pd.concat(frames_dict[motion], axis =1, join="outer")
        else: #Here we account for the dates being slightly jumbled 
            # We find the index that covers all the data
            full_index = frames_dict[motion][0].index
            for frame in frames_dict[motion][1:]:
                full_index = full_index.union(frame.index)
            errors = []
            index = 0  # Start from the end of the list
            gap = 1
            #Prev things are the things previously in time but ahead in the iteration
            while index + gap < len(full_index):  # Ensure index is within the bounds of the list
                date = full_index[index]
                next_date = full_index[index+gap]
                if next_date - date < pd.Timedelta(days = months_dict[date.month] -4):  #If the dates are too close to each other
                    errors.append(next_date)
                    gap += 1  # Increase gap to skip further ahead
                else:
                    index += gap  # Move to the next item 
                    gap = 1
            dates_to_remove_set = set(errors)
            new_index = pd.DatetimeIndex([date for date in full_index if date not in dates_to_remove_set]) 
            
            frame_list = []
            for frame in frames_dict[motion]:
                frame = frame.reindex(new_index, method='nearest', limit=10)
                frame_list.append(frame)
            if frame_list != []:
                df[motion] = pd.concat(frame_list, axis =1, join="outer")
            else:
                continue
        df[motion] = df[motion][comp.extreme_start[motion]:comp.extreme_end[motion]]
        #If units are necessary
        # columns_multiindex = pd.MultiIndex.from_tuples([(col, unit) for col, unit in zip(df.columns, unit_list)],names=['Variable', 'Unit'])
        # df.columns = columns_multiindex
        df[motion].attrs["units"] = unit_dict[motion]
        df[motion].attrs["category"] = catg
        df[motion].attrs["missing"] = False
        for unit in unit_dict[motion]:
            if unit == "missing":
                df[motion].attrs["missing"] = True
                break
    return df

#Initializes and appends the stock object
async def async_task(ticker, client, semaphore_edgar, semaphore_yahoo, measures):
    # Measures are used to get the date when all the financial info is available
    print(f"Loading {ticker}")
    try:
        stock = Stock(ticker, measures)
        if stock.success == "del":
            return (ticker, "del")
        # successful_sic = await stock.async_init(client,semaphore_edgar,measures)
        # if successful_sic == "del":
        #     return (ticker, "del")
        if stock.success[0] or stock.success[1]:
            print(f"Price pinging {ticker}$")
            succesful_price = await asyncio.wait_for(stock.price_init(semaphore_yahoo), timeout=500)
        else:
            succesful_price = 0
        with open(f'..\\companies\{ticker}.pkl', 'wb') as file:
            pickle.dump(stock,file)
        success = stock.success
        del stock
        #Return (ticker, availability of data, availability of price)
        print(f"||Done {ticker}|| = {success}")
        return (ticker, [success, succesful_price])
    except asyncio.TimeoutError as ex:
        print(f"Timeout {ex}")
        return (ticker, [(0,0),0])
    except asyncio.CancelledError as ex:
        print(f"Canceled {ex}")
        return (ticker, [(0,0),0])
    # except BaseException as e:
    #     print(f"||Done {ticker}|| = {success}")
    #     return (ticker, [(0,0),0])
    except Exception as e:
        print(f"Async_task failed because: {e}")
        print(f"||Done {ticker}|| = {success}")
        return (ticker, [(0,0),0])
#Get the success rate for the api call
def success_rate(availability_list):
    static_success = 0
    dynamic_success = 0
    yahoo_success = 0
    for i in availability_list:
        try:
            ticker, available = i
            if available != "del":
                static_success += available[0][0]
                dynamic_success += available[0][1]
                yahoo_success += available[1]
            else:
                static_success += 1
                dynamic_success += 1
                yahoo_success += 1
        except Exception:
            continue
    try:
        static_success = static_success/len(availability_list)
        dynamic_success = dynamic_success/len(availability_list)
        yahoo_success = yahoo_success/len(availability_list)
        print(f"Static success rate: {static_success}")
        print(f"Dynamic success rate: {dynamic_success}")
        print(f"Yahoo success rate: {yahoo_success}")
    except ZeroDivisionError:
        print("No list to analyze")
            
#Function to call again for missing data
def ticker_fill(company_frames_availability):
    ticker_list = []
    for ticker, available in company_frames_availability.items():
        success, price = available
        if success == "del":
            continue
        (static, dynamic) = success
        if static and dynamic and price:
            continue
        else:
            ticker_list.append(ticker)
    print(f"Calling {len(ticker_list)} companies")
    return ticker_list

def get_label_columns(comp, intervals=None, dividends=False):
    if comp.foreign:
        print("Cannot use ADR for foreign companies")
        return pd.DataFrame()
    static = comp.share_availability[0]
    dynamic = comp.share_availability[1]
    static_have = False
    dynamic_have = False
    if static:
        static_frame, unit = comp.fact("EntityCommonStockSharesOutstanding", intervals = intervals, static_tolerance=pd.Timedelta(days=10000), lookbehind = 1, share_filter = 1)
        if not static_frame.empty:
            static_have = True
    if dynamic:
        dynamic_frame, unit = comp.fact("WeightedAverageNumberOfSharesOutstandingBasic", intervals = intervals, static_tolerance=pd.Timedelta(days=10000), lookbehind = 1, forced_static = 1, share_filter = 1)
        if dynamic_frame.empty:
            dynamic_frame, unit = comp.fact("WeightedAverageNumberOfDilutedSharesOutstanding", intervals = intervals, static_tolerance=pd.Timedelta(days=10000), lookbehind = 1, forced_static = 1, share_filter = 1)
            if not dynamic_frame.empty:
                dynamic_have = True
        else:
            dynamic_have = True
    if static_have and dynamic_have:
        dynamic_frame = frame_rename(dynamic_frame, "EntityCommonStockSharesOutstanding")
        frame = static_frame.combine_first(dynamic_frame)
    elif static_have:
        frame = static_frame
    elif dynamic_have:
        frame = frame_rename(dynamic_frame, "EntityCommonStockSharesOutstanding")
    else:
        print("Neither share method is available")
        return pd.DataFrame()
    frame.ffill(inplace=True)
    frame["MarketCap"] = frame["EntityCommonStockSharesOutstanding-0"] * comp.price["close"]
    if dividends: #This is really kind of obsolete, ist just a fun way to get the real historical market cap
        if not comp.dividends.empty:
            comp.dividends["multiplier"] = (comp.dividends["dividend"]/comp.price["close"] + 1)
            multipliers = comp.dividends["multiplier"]
            multipliers.dropna(inplace=True)
            adjustments = pd.Series(1, index=frame.index.union(multipliers.index))  
            adjustments.loc[multipliers.index] = multipliers
            frame['MarketCap'] *= adjustments[::-1].cumprod()[::-1]
    #Add the dividends as a feature because they always influence the price irl for some reason
    dividends = pd.Series(0, index=frame.index.union(comp.dividends.index), dtype='float64')  
    if not comp.dividends.empty:
        dividends.loc[comp.dividends["dividend"].index] = comp.dividends["dividend"]
    #Add the splits as a feature because they always influence the price irl for some reason
    splits = pd.Series(1, index=frame.index.union(comp.splits.index), dtype='float64')
    if not comp.splits.empty:  
        splits.loc[comp.splits["splitRatio"].index] = comp.splits["splitRatio"] #Here we include the splits ratio, where there is no split there is 1 
    frame["splits"] = splits
    frame["dividend"] = dividends
    frame["close"] = comp.price["close"]
    frame.ffill(inplace=True)
    return frame        