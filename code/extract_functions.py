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

def unitrun(dict, ticker, all=False, debug=False):
    if all:
        unit_list = all_units
    else:
        unit_list = valid_units
    for unit in unit_list:
        try:
            return dict[unit], unit 
        except KeyError:
            continue
    print(f"No unit available for {ticker}")
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

    print(f"{ticker}:{(duplicate_count/(filtered_count+duplicate_count))*100}%")
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
        missing = False
        category = get_category(self.sic)
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
                pathways = recursive_date_gather(self, measure)
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
        #remove duplicates
        needed_measures = list(set(needed_measures))
        self.initialized_measures = standard_measures
        #Change the strings to datetime for the initialized measures
        #Flatten batches into a single DataFrame with an identifier
        all_data = []
        for measure in needed_measures:
            if measure in deprecate_conversion:
                measure = deprecate_conversion[measure]
            datapoints, unit = unitrun(self.data[measure]["units"], self.ticker)
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
        self.fullprice = await yahoo_fetch(self.ticker, self.extreme_start, self.extreme_end, semaphore, RETRIES, START_RETRY_DELAY)
        if type(self.fullprice) == int:
            return 0
        Price = self.fullprice[["close", "adjclose"]].copy()
        self.price = Price.ffill().bfill()
        return 1 
    
    def fact(self, measure, intervals = None, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5, annual=False, reshape_approx=False, date_gather=False):
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
                return "del"
            reshaped, intervals, dynamic = reshape(measure, point_list, self.ticker, converted=False)
            return intervals
        if measure in self.converted_data:
            data= self.converted_data[measure]
            converted= True
        elif measure in irrelevants[get_category(self.sic)]:
            base_dates = pd.date_range(start=self.extreme_end["static"], end=self.extreme_start["static"], freq=-row_delta)
            frame = pd.DataFrame(index =base_dates, columns = [f"{measure}-{i}" for i in range(0,lookbehind)])
            frame = frame.infer_objects()
            frame.fillna(0,inplace = True)
            frame = frame.iloc[::-1]
            return frame, "ignored"
        elif measure in self.data:
            data = self.data[measure]
            converted = False
        else:
            # print(f"{self.ticker}: Data not converted or available for {measure}")
            return pd.DataFrame(), None
        #Get the index dates for the datpoints for measure
        #If the measure doesn't make sense for the company's category, replace all of the datapoints with 0
        reshaped, all_intervals, dynamic = reshape(measure, data, self.ticker, annual, reshape_approx, converted=converted) # _ is intervals, if the data has not been converted in the init, it will be here
        if intervals == None:
            intervals = all_intervals 
        for k ,interval in enumerate(intervals): #To save compute we only get the data that we really need, could be intervals with gaps
            data_start, data_end = interval
            if dynamic:
                row_delta = dynamic_row_delta
                column_delta = pd.Timedelta(days=95)
            tolerance = dynamic_tolerance * int(dynamic) + static_tolerance * (int(not dynamic))
            data_start = data_start - lookbehind * column_delta - pd.Timedelta(days=5) #To account for the lookbehind, it does not matter if it is way too much, it will just turn out blank and we can handle it later
            if not dynamic: #static 
                dates = list(reshaped.keys())
                base_dates = pd.date_range(start=data_end, end=data_start, freq=-row_delta) #Inverted start and end to keep the correct order
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
                rows = [zip(timestamps[0:lookbehind], [None] + timestamps[0:lookbehind-1])]
                for i in range(lookbehind+1, index_length):
                    # Extract the window of n elements
                    window = zip(timestamps[i-lookbehind:i], timestamps[i-1-lookbehind:i-1])
                    # Append the window to the rows list
                    rows.append(window)

                # Convert the list of rows into a DataFrame
                row_indexes = pd.DataFrame(rows, index = timestamps[lookbehind:])
                frame_dict = {row_index:[] for row_index in timestamps[lookbehind:]}
                
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
                total_frame = frame
                total_frame = pd.DataFrame(columns=[f"{measure}-{i}" for i in range(lookbehind)])
            elif k == 0:
                total_frame = frame
            elif k != 0:
                total_frame = total_frame.combine_first(frame)
        total_frame = total_frame.iloc[::-1]
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

def operate(comp, frames, frame_names, operation, measure_name, approx, approx_needed = 0):
    #Rename the frames so we can easily use them and check if they are empty to prevent errors
    frames_and_names = [(frame_rename(frame, measure_name),frame_name) for frame, frame_name in zip(frames,frame_names) if frame.empty == False]
    frames, frame_names = zip(*frames_and_names)
    if operation == "sub" and len(frames) != 2:
        print("Need two frames to sub, the empty check could have triggered")
    static = comp.measures_and_intervals["static"].get(frame_names[0], False)
    if static == False:
        dynamic = comp.measures_and_intervals["dynamic"].get(frame_names[0], False)
        if dynamic == False:
            print("Operate: frame not in measures_and_intervals")
        else:
            motion = "dynamic"
    else:
        motion = "static"
    #First we find the intervals for the data that we have 
    intervals_list = []
    for name in frame_names:
        intervals = comp.measures_and_intervals[motion][name]
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
        overlap_intervals = calculate_overlap(intervals_list)
        total_frame = pd.DataFrame()
        for overlap in overlap_intervals:
            overlap_start, overlap_end = overlap
            # This makes sure that the real overlap is included
            # extended_start = overlap_start - pd.Timedelta(days=5)
            # extended_end = overlap_end + pd.Timedelta(days=5)
            extended_start = overlap_start
            extended_end = overlap_end
            temp_frames = [frame[extended_end:extended_start] for frame in frames] #Start is switched with end because our frame are inverted
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
        return total_frame[::-1]
    
    else: #We keep residuals here 
        #Make a total index that we will use as the canvas for all the operations
        all_indices = pd.concat([pd.Series(frame.index) for frame in frames if not frame.index.empty])
        index_combined = pd.Index(all_indices.dropna().unique())
        total_index = index_combined.sort_values()
        differences = total_index.to_series().diff()
        differences.iloc[0] = pd.Timedelta(days=91) #To keep the first value 
        filtered_index = total_index[differences > pd.Timedelta(days=5)] #We do this so we they are all spaced enough apart 
        starting_frame = pd.DataFrame(0, index=filtered_index, columns =frames[0].columns) #Make it the same as previous frames
        starting_frame = starting_frame[::-1]
        starting_frame["checker"] = 0
        for frame in frames:
            frame["checker"] = 1
            frame.apply(lambda row: 0 if row.isnull().any() else 1) #You can implement fractions
            frame.fillna(0, inplace = True)
        if operation == "add":
            for frame, intervals in frames_and_intervals:
                for interval in intervals:
                    start, end = interval
                    extended_start = start - pd.Timedelta(days=5)
                    extended_end = end + pd.Timedelta(days=5)
                    reasign_index = starting_frame[extended_end:extended_start].index
                    temp_frame = frame[extended_end:extended_start]
                    temp_frame.index = reasign_index
                    starting_frame = starting_frame.add(temp_frame, fill_value=0)[::-1]
            final_frame = starting_frame[starting_frame["checker"] >= approx_needed] #Change this cause we want the nans and a continous interval
            final_frame.drop(columns=["checker"])
            return final_frame
        else:
            print("Only add is possible with approx")

def path_selector(comp, measure, path, intervals = None, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5 , annual=False, reshape_approx = False):
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
            value, unit = comp.fact(root, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            value = frame_rename(value,measure)
            return value, unit
        #Concat path
        elif root == "concat":
            fledgling, get_interval = tree[0]
            if intervals != None:
                new_intervals = calculate_overlap([intervals, [get_interval]])
            else:
                new_intervals = [get_interval]
            concat_frame, unit = path_selector(comp, measure, {measure:fledgling}, new_intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            for sapling in tree[1:]:
                fledgling, get_interval = sapling
                if intervals != None:
                    new_intervals = calculate_overlap([intervals, [get_interval]])
                else:
                    new_intervals = [get_interval]
                frame, unit = path_selector(comp, measure, {measure:fledgling}, new_intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
                if frame.empty:
                    continue
                frame.columns = concat_frame.columns
                concat_frame = concat_frame.combine_first(frame)
            concat_frame = frame_rename(concat_frame, measure)
            return concat_frame, unit  #The units are the same...
        #Add path
        elif root == "add":
            values = []
            for sprout, tree_part in tree.items():
                add_value, unit = path_selector(comp, sprout ,{sprout:tree_part}, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
                values.append(add_value)
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
            result_frame = frame_rename(result_frame,measure)
            return result_frame, unit
            
        elif root == "sub":
            larch = list(tree.items())
            add, unit = path_selector(comp, larch[0][0], {larch[0][0]:larch[0][1]}, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            sub, unit = path_selector(comp, larch[1][0], {larch[1][0]:larch[1][1]}, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            #Get all the possible indexes for the frame
            values = [add, sub]
            add, sub = add.align(sub, join="outer", axis=0)
            result_frame = pd.DataFrame(index=add.index)
            for i, col in enumerate(add.columns):
                result_frame[f'{measure}-{i}'] = add[col]  # Initialize with add's columns
            for i, col in enumerate(sub.columns):
                result_frame[f'{measure}-{i}'] = result_frame[f'{measure}-{i}'].sub(sub[col])
            result_frame = frame_rename(result_frame,measure)
            return result_frame,unit
        
        elif root == "approx_add":
            values = []
            for sprout, tree_part in tree.items():
                add_value, unit = path_selector(comp, sprout ,{sprout:tree_part}, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
                #Specifically for approx
                add_value.fillna(0, inplace =True)
                values.append(add_value)
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
            result_frame = frame_rename(result_frame,measure)
            #Specifically for approx
            result_frame.replace(0,np.nan, inplace=True)
            return result_frame, unit

        #serves as the measure conversion part
        else:
            return path_selector(comp, root, tree, intervals, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)

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
        if overlap == None:
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
        concat_paths.append((first_path,(first_start,first_end)))
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
                        if route in concat_paths: #Prevent further tests and just add it in there, because we have already used it, the check means that we used an interval through the same method
                            break
                    elif end == best_end and (len(needed)<len(best_needed)):
                        idx = k
                        best = route
                        best_start = start
                        best_end = end
                        best_needed = needed
            if best == None: #since that means there are no other eligible intervals to add 
                if appended_flag == False:
                    intervals.append((total_start, current_end)) #Add the closed interval we got 
                first_path, (first_start, first_end), first_needed = sorted_paths[idx+1] ###
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
                    concat_paths.append((first_path,(first_start,first_end)))
                    needed_measures += first_needed
                    total_start = first_start
                    current_end = first_end
                    appended_flag = False
                else:
                    idx += 1 #Go to the next iteration
                    appended_flag = True
            else: 
                current_end = best_end
                concat_paths.append((best[0], (best_start,best_end)))
                needed_measures += best_needed
        if intervals == []: #If we also used the last interval we still have to append it 
            intervals.append((total_start, current_end))

        return ({"concat":concat_paths}, intervals, list(set(needed_measures)))
    
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

#Used: {"conversion":["bla", "blabla"], "addition":[blabla]}
def recursive_date_gather(comp, measure, depth=0, path_date=None, approx = True, used=None, printout =False):
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
        if used is None:
            used = {"conversion":[],"approx_conversion":[],"add":[],"approx_add":[],"sub":[]}
    if depth>5:
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
        if measure not in used["conversion"]:
            used["conversion"].append(measure)
            for replacement in measure_conversion[measure]:
                path = recursive_date_gather(comp, replacement, depth+1, path_date, approx, used, printout)
                # path = path_finder(path)
                if path != None:
                    paths.append(({replacement:path[0]},path[1],path[2]))

    if measure in additive_conversion:
        if measure not in used["add"]:
            used["add"].append(measure)
            branches = []
            parts = []
            if additive_conversion[measure] != []:
                abort = False
                for part in additive_conversion[measure]:
                    if abort == False:
                        path = recursive_date_gather(comp, part,depth+1,path_date, approx, used, printout) 
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
        if measure not in used["sub"]:
            used["sub"].append(measure)
            path_add = recursive_date_gather(comp,subtract_conversion[measure][0],depth+1, path_date, approx, used, printout)
            path_sub = recursive_date_gather(comp,subtract_conversion[measure][1],depth+1, path_date, approx, used, printout)
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
            if measure not in used["approx_conversion"]:
                used["approx_conversion"].append(measure)
                for replacement in approximate_measure_conversion[measure]:
                    path = recursive_date_gather(comp, replacement, depth+1, path_date, approx, used, printout)
                    # path = path_finder(path)
                    if path != None:
                        paths.append(({replacement:path[0]},path[1],path[2]))     
        if measure in approximate_additive_conversion:
            if measure not in used["approx_add"]:
                used["approx_add"].append(measure)
                branches = []
                parts = []  #We need to record which parts we actuall used
                optionals = 0 #To calc the percentage of optionals used
                approxes, parts_needed = approximate_additive_conversion[measure]
                if approxes != []:
                    abort = False
                    for part in approxes:
                        if abort == False:
                            path = recursive_date_gather(comp, part,depth+1,path_date, approx, used, printout) 
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
                            paths.append(({measure:{"add":{part:path[i] for i,part in enumerate(parts)}}}, interval, needed))
                        else:
                            paths.append(({"add":{part:path[i] for i,part in enumerate(parts)}}, interval, needed))            
    if depth ==0:
        if path_date["del"] == True:
            return path_date
        paths = path_finder(paths)
        path_date["paths"] = paths
        return path_date    
    
    if paths == []:
        return None
    paths = path_finder(paths)
    return paths

def get_category(sic):
    for category, number_ranges in categories.items():
        for number_range in number_ranges:
            if number_range[0]<=sic<=number_range[-1]:
                return category
    return "Uncategorized"

#(comp, measure, depth=0, approx = True, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5 , annual=False, printout =False, date_gather= False
def acquire_frame(comp, measures:dict, available, indicator_frame, reshape_approx= True, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365), static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91),  lookbehind =5, annual=False):
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
        for measure in measures[motion]:
            if measure not in comp.missing[motion]:
                print(f"{comp.ticker} Getting {measure}")
                print(comp.measure_paths[motion][measure])
                intervals = comp.measures_and_intervals[motion][measure]
                data, unit = path_selector(comp, measure, comp.measure_paths[motion][measure], intervals, row_delta , column_delta , static_tolerance, dynamic_row_delta,dynamic_tolerance, lookbehind, annual, reshape_approx)
                data.name = measure
                frames_dict[motion].append(data)
                unit_dict[motion].append(unit)
        if frames_dict[motion] == []:
            break
        df[motion] = pd.concat(frames_dict[motion], axis =1, join="outer")
    #If units are necessary
    # columns_multiindex = pd.MultiIndex.from_tuples([(col, unit) for col, unit in zip(df.columns, unit_list)],names=['Variable', 'Unit'])
    # df.columns = columns_multiindex
    #Economic indicators 
    # indicator_frame = indicator_frame.reindex(comp.date_range)
    # indicator_frame = indicator_frame.ffill().bfill()
    # df = df.join(indicator_frame, how="left")
        df[motion] = df[motion].join(comp.price, how= "left")
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
    stock = Stock(ticker, measures)
    if stock.success == "del":
        return (ticker, "del")
    # successful_sic = await stock.async_init(client,semaphore_edgar,measures)
    # if successful_sic == "del":
    #     return (ticker, "del")
    if stock.success[0] or stock.success[1]:
        print(f"Price pinging {ticker}$")
        succesful_price = await stock.price_init(semaphore_yahoo)
    else:
        succesful_price = 0
    with open(f'..\\companies\{ticker}.pkl', 'wb') as file:
        pickle.dump(stock,file)
    success = stock.success
    del stock
    #Return (ticker, availability of data, availability of price)
    print(f"||Done {ticker}||")
    return (ticker, [success, succesful_price])

#Get the success rate for the api call
def success_rate(availability_list):
    static_success = 0
    dynamic_success = 0
    yahoo_success = 0
    for i in availability_list:
        ticker, available = i
        if available != "del":
            static_success += available[0][0]
            dynamic_success += available[0][1]
            yahoo_success += available[1]
        else:
            static_success += 1
            dynamic_success += 1
            yahoo_success += 1
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
        (static, dynamic), price = available
        if static and dynamic and price:
            continue
        else:
            ticker_list.append(ticker)
    return ticker_list