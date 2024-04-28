FACTS_PATH = r"C:\Edgar_data"
SUBMISSIONS_PATH = r"C:\Submissions_data"

#Unavailable stuff
with open(r'other_pickle\unavailable.json', 'r') as file:
        Unavailable_Measures = json.load(file)

#Lookup table for the undeprecated version of a measure
with open(r"other_pickle\deprecated_to_current.json", "r") as file:
    deprecate_conversion = json.load(file)
    file.close()

#Categories
with open(r"categories\categories.json", "r") as file:
    categories = json.load(file)

#Irrelevants
with open(r"categories\category_measures_irrelevant.json") as file:
    irrelevants = json.load(file)

#Make sure the necessary folders exist
for category, num_range in categories.items():
    os.makedirs(f"companies_data\static\{category}", exist_ok=True)
    os.makedirs(f"companies_data\dynamic\{category}", exist_ok=True)
    os.makedirs(f"companies_data_missing\{category}", exist_ok=True)

for path in ["checkout", "companies", "companies_data", "companies_data_missing", "units-checkout"]:
    os.makedirs(path, exist_ok=True)

with open("categories\category_measures.json", "r") as file:
    category_measures = json.load(file)

#The first entry date into the EDGAR database
START = datetime.strptime('1993-01-01', r"%Y-%m-%d")

#Easily load a company into a variable
def comp_load(ticker):
    with open(f"companies\{ticker}.pkl", "rb")as file:
        company = pickle.load(file)
    return company

def example_save(data, name):
    with open(f"examples/{name}.json", "w") as file:
        json.dump(data,file,indent=1)
    print("Saved")
    
#Manually figure out which measure is used with some company
def company_wordsearch(ticker, word):
    with open(f"companies\{ticker}.pkl", "rb")as file:
        company = pickle.load(file)
    data = company.data
    compdict = {}
    for key,value in data.items():
        compdict[key] = value["description"]

    matching_elements  ={}
    for name, desc in compdict.items():
        if word.lower() in name.lower():   
            matching_elements[name] = desc 
    with open(f"checkout\{ticker}.json", "w") as file:
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
        with open(f"units-checkout\\{ticker}.json", "w") as file:
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

with open(r"other_pickle\cik.json","r") as file:
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
            self.start_dates = {}
            self.end_dates = {}
            self.measures_and_dates = {}
            self.success = self.time_init(standard_measures)
        except FileNotFoundError:
            self.success = "del"
        # except Exception as e:
        #     print(f"During Initialization: {e}")

    def time_init(self, standard_measures, static_start_threshold = 1, static_end_threshold = 1, dynamic_start_threshold = 1, dynamic_end_threshold = 1):
        start_thresholds = {"static": static_start_threshold, "dynamic": dynamic_start_threshold}
        end_thresholds = {"static": static_end_threshold, "dynamic": dynamic_end_threshold}
        #Also serves as a filter for the companies with wrong currencies
        #Check to see if already initialized with the measures
        missing = False
        category = get_category(self.sic)
        self.start_year = {}
        self.end_year = {}
        self.date_range = {}
        measure_paths = {"static":{}, "dynamic":{}}
        needed_measures = []
        measures_and_dates = {"static":[], "dynamic":[]}
        date_dict = {"static":[], "dynamic":[]}
        #Both static and dynamic time init
        missing_measures = {"static":[], "dynamic":[]}
        irrelevant_measures = []
        success = {"static":0, "dynamic":0}
        flags = {"static":0, "dynamic":0}
        for motion in ["static", "dynamic"]:
            for measure in standard_measures[motion]:
                if not measure in self.initialized_measures[motion]:
                    missing = True
                    break
            if not missing:
                #We do this because even if initialized with the same measures, thresholds could be different
                start_index = math.ceil(start_thresholds[motion]*len(self.start_dates[motion])) -1 
                end_index = -math.ceil(end_thresholds[motion]*len(self.end_dates[motion])) #While going through the list you lose data instead of gain so the index is different
                self.start_year[motion] = pd.to_datetime(self.start_dates[motion][start_index], format=r"%Y-%m-%d")
                self.end_year[motion] = pd.to_datetime(self.end_dates[motion][end_index], format=r"%Y-%m-%d")
                self.date_range[motion] = pd.date_range(start=self.start_year[motion], end=self.end_year[motion])
                flags[motion] = 1
                continue
            # Get all the constituent measures needed through recursivefact
            for measure in standard_measures[motion]:
                print(f"Getting {measure}")
                #Make a copy.deepcopy if not working 
                if measure in deprecate_conversion:
                    measure = deprecate_conversion[measure]
                pathways = recursive_date_gather(self, measure)
                if pathways["del"]:
                    return "del"
                # needed_measures += pathways["needed"]
                del pathways["del"]
                if pathways["paths"] != None:
                    measure_paths[motion].update({measure: (pathways["paths"][0], pathways["paths"][1])})
                    needed_measures += pathways["paths"][2]
                    date_dict[motion].append(pathways["paths"][1])
                    measures_and_dates[motion].append((measure, pathways["paths"][1]))
                    # if pathways["ignored"] != []:
                    #     # print(f"{self.ticker}:{pathways}:{measure}")
                    #     irrelevant_measures += pathways["ignored"]
                else:
                    missing_measures[motion].append(measure)
            if measure_paths[motion] == {}:
                return "del"  
            overlap = calculate_overlap(date_dict[motion])   
            # # start_dates, end_dates = map(lambda x: sorted(list(x)),zip(*date_dict[motion]))
            # start_index = math.ceil(start_thresholds[motion]*len(start_dates)) -1 
            # end_index = -math.ceil(end_thresholds[motion]*len(end_dates))
            # self.start_dates[motion] = start_dates
            # self.end_dates[motion] = end_dates
            # self.start_year[motion] = pd.to_datetime(start_dates[start_index], format=r"%Y-%m-%d")
            # self.end_year[motion] = pd.to_datetime(end_dates[end_index], format=r"%Y-%m-%d")
            # self.date_range[motion] = pd.date_range(start=self.start_year[motion], end=self.end_year[motion])
            # if self.end_year[motion]< self.start_year[motion]:
            #     success[motion] = 0
            # else:
            #     success[motion] = 1
            # self.measures_and_dates = measures_and_dates
        return (1,1)
        if flags["static"] and flags["dynamic"]:
            return (flags["static"], flags["dynamic"])
        #measure paths look like: 
        self.paths = measure_paths
        self.missing = missing_measures
        self.ignored = irrelevant_measures
        #remove duplicates
        needed_measures = list(set(needed_measures))
        # for i in self.ignored:
        #     needed_measures.remove(i)
        self.initialized_measures = standard_measures

        #Change the strings to datetime for the initialized measures
        # Step 1: Flatten batches into a single DataFrame with an identifier
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
        self.fullprice = await yahoo_fetch(self.ticker,self.start_year, self.end_year, semaphore, RETRIES, START_RETRY_DELAY)
        if type(self.fullprice) == int:
            return 0
        Price = self.fullprice[["close", "adjclose"]].copy()
        # Price = Price.reindex(self.date_range)
        self.price = Price.ffill().bfill()
        return 1 
    
    def fact(self, measure, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5, annual=False, reshape_approx=False, date_gather=False):
        """  
        If date_gather, then it returns a dataframe to allow recursive_fact to get the date.
        Returns a dataframe that has rows indexed row_delta away, with lookbehind columns that are column_delta away.
        If the data is dynamic then the row and column deltas are fixed.
        Dynamic tolerance is how much into the future the price we are predicting is.
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
            reshaped, intervals, dynamic = reshape(measure, point_list)
            return intervals
        
        #Get the index dates for the datpoints for measure
        if measure in self.converted_data:
            data= self.converted_data[measure]

        #If the measure doesn't make sense for the company's category, replace all of the datapoints with 0
        elif measure in irrelevants[get_category(self.sic)]:
            base_dates = pd.date_range(start=self.end_year["static"], end=self.start_year["static"], freq=-row_delta)
            frame = pd.DataFrame(index =base_dates, columns = [f"{measure}-{i}" for i in range(0,lookbehind)])
            frame = frame.infer_objects()
            frame.fillna(0,inplace = True)
            frame = frame.iloc[::-1]
            return frame, "ignored"
        else:
            # print(f"{self.ticker}: Data not converted or available for {measure}")
            return pd.DataFrame(), None
        reshaped, intervals, dynamic = reshape(measure, data, annual, reshape_approx)
        if dynamic:
            motion = "dynamic"
        else:
            motion = "static"
        if dynamic == True:
            row_delta = dynamic_row_delta
            column_delta = pd.Timedelta(days=95)
        tolerance = dynamic_tolerance * int(dynamic) + static_tolerance * (int(not dynamic))
        if motion == "static":
            dates = list(reshaped.keys())
            base_dates = pd.date_range(start=self.end_year[motion], end=self.start_year[motion], freq=-row_delta)
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
            return frame, None
        frame = frame.iloc[::-1]
        frame.columns = [f"{measure}-{i}" for i in range(0,lookbehind)]
        return frame, "Some_unit"   
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

def operate(frames, operation, approx):
    if not approx: 
        overlap_start = max([frame.index.min() for frame in frames])
        overlap_end = min([frame.index.max() for frame in frames])
        # This makes sure that the real overlap is included
        extended_start = overlap_start - pd.Timedelta(days=5)
        extended_end = overlap_end + pd.Timedelta(days=5)

        frames = [frame[extended_start:extended_end] for frame in frames]
        reasign_index = frames[0].index 
        frames = [frame.reset_index(drop=True) for frame in frames]
        if operation == "sub":
            resulting_frame = frames[0] - frames[1]

        if operation == "add":
            resulting_frame = frames[0]
            for frame in frames[1:]:
                resulting_frame += frame

        resulting_frame.index = reasign_index
    else: #We keep residuals here 
        if operation == "add":
            pass 

def path_selector(comp, measure, path, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365),static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91), lookbehind =5 , annual=False, reshape_approx = False):
    """
    Takes in the desired path to the data and outputs the data
    New recursive_fact
    There is no approx since the approximity of the path is determined by recursive_date_gather
    """
    #[({'Assets': True}, ('2008-09-27', '2023-12-30')), ({'LiabilitiesAndStockholdersEquity': True}, ('2008-09-27', '2023-12-30')), ({'add': {'AssetsCurrent': True, 'AssetsNoncurrent': {'sub': {'Assets': True, 'AssetsCurrent': True}}}}, ('2008-09-27', '2023-12-30'))]
    #[{'Assets': (True, ('2008-09-27', '2023-12-30'))]}, ({'LiabilitiesAndStockholdersEquity': (True, ('2008-09-27', '2023-12-30'))}), ({'add': {'AssetsCurrent': (True,('2008-09-27', '2023-12-30')), 'AssetsNoncurrent': {'sub': {'Assets': True, 'AssetsCurrent': True}}}})]
    for root, tree in path.items():
        if tree == True:              
            value, unit = comp.fact(root, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            value = frame_rename(value,measure)
            return value, unit
        #Concat path
        elif root == "concat":
            concat_frame, unit = path_selector(comp, measure, {measure:tree[0]}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            for sapling in tree[1:]:
                frame, unit = path_selector(comp, measure, {measure:sapling}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
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
                add_value, unit = path_selector(comp, sprout ,{sprout:tree_part}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
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
            add, unit = path_selector(comp, larch[0][0], {larch[0][0]:larch[0][1]}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
            sub, unit = path_selector(comp, larch[1][0], {larch[1][0]:larch[1][1]}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
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
                add_value, unit = path_selector(comp, sprout ,{sprout:tree_part}, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)
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
            return path_selector(comp, root, tree, row_delta, column_delta, static_tolerance, dynamic_row_delta, dynamic_tolerance, lookbehind, annual, reshape_approx)

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
                    if start <= current_end and end > current_end:
                        idx = k
                        best = route
                        best_start = start
                        best_end = end
                        best_needed = needed
                    else:
                        continue
                elif start <= current_end:
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
                        if start <= first_end and end > first_end or (end == first_end and (len(needed)<len(first_needed))): #or (len(needed)==len(best[2]) and analyze_structure(route) < analyze_structure(best)
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

def recursive_date_gather(comp, measure, depth=0, path_date=None, blind = False, approx = True, printout =False):
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
    if path_date is None and depth==0:
        path_date = {"del": False, "paths":[], "ignored":[]}
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
            paths.append(({measure:True},(pd.Timestamp(year=1970, month=1, day=1), pd.Timestamp.now()),[])) #Just to make path seeking better
        else:   
            paths.append((True,(pd.Timestamp(year=1970, month=1, day=1), pd.Timestamp.now()),[]))
        path_date["ignored"].append(measure)
    elif dates != None:
        if depth==0:
            paths.append(({measure:True},dates,[measure])) #Just to make path seeking better
        else:   
            paths.append((True,dates,[measure]))
        
    if measure in measure_conversion:
        for replacement in measure_conversion[measure]:
            path = recursive_date_gather(comp, replacement, depth+1, path_date, blind, approx, printout)
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
                    path = recursive_date_gather(comp, part,depth+1,path_date, True, approx, printout) 
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
        path_add = recursive_date_gather(comp,subtract_conversion[measure][0],depth+1, path_date, True, approx, printout)
        path_sub = recursive_date_gather(comp,subtract_conversion[measure][1],depth+1, path_date, True, approx, printout)
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
                path = recursive_date_gather(comp, replacement, depth+1, path_date, blind, approx, printout)
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
                        path = recursive_date_gather(comp, part,depth+1,path_date, True, approx, printout) 
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
def acquire_frame(comp, measures:dict, available, indicator_frame, static_start_threshold = 1, static_end_threshold = 1, dynamic_start_threshold = 1, dynamic_end_threshold = 1, reshape_approx= True, row_delta = pd.Timedelta(days=1), column_delta = pd.Timedelta(days=365), static_tolerance=pd.Timedelta(days =0), dynamic_row_delta=pd.Timedelta(days=1), dynamic_tolerance=pd.Timedelta(days=91),  lookbehind =5, annual=False):
    #Get a dataframe from the saved data of some stock 
    #Returns 0 in all the columns where data is missing
    comp.time_init(measures, static_start_threshold, static_end_threshold , dynamic_start_threshold , dynamic_end_threshold)
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
                print(comp.paths[motion][measure][0])
                data, unit = path_selector(comp, measure, comp.paths[motion][measure][0], row_delta , column_delta , static_tolerance, dynamic_row_delta,dynamic_tolerance, lookbehind, annual, reshape_approx)
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
    with open(f'companies\{ticker}.pkl', 'wb') as file:
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