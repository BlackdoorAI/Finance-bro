import numpy as np
import pandas as pd
import os
import pickle
from conversions import *


def timediff(a,b):
    return abs((a-b).days)

def reshape(measure, datapoint_list, ticker, annual = False, approx = False, converted = False, use_precompute = True, forced_static = 0):
    #Reshapes the datapoint list so that its indexed by end and each item retains its attrs
    #Designed to be used after data is converted to datetime
    #If we have it precomputed we just use it
    if use_precompute:
        if os.path.exists(f"C:\\Programming\\Python\\Finance\\EDGAR\\reshaped\\{ticker}\\{measure}.pkl"):
            with open(f"C:\\Programming\\Python\\Finance\\EDGAR\\reshaped\\{ticker}\\{measure}.pkl", "rb") as file:
                trinity = pickle.load(file)
                return trinity
    dynamic = True
    if "start" not in datapoint_list[0]: 
        if measure not in dynamic_fuckers:
            dynamic = False
    elif pd.isna(datapoint_list[0]["start"]):
        dynamic = False
    if not converted:
        df = pd.DataFrame(datapoint_list)
        df['filed'] = pd.to_datetime(df['filed'], format='%Y-%m-%d')
        df['end'] = pd.to_datetime(df['end'], format='%Y-%m-%d')
        if dynamic:
            df['start'] = pd.to_datetime(df['start'], format='%Y-%m-%d', errors='coerce')
        datapoint_list = df.to_dict("records")
    if forced_static:
        reshaped_data = {}
        for item in datapoint_list:
            date = item["end"]
            if date not in reshaped_data:
                reshaped_data[date] = []
            reshaped_data[date].append({
                "val": item["val"],
                "start": item.get("start"),
                "filed": item["filed"],
            })
    elif dynamic ==False:
        reshaped_data = {}
        for item in datapoint_list:
            date = item["end"]
            if date not in reshaped_data:
                reshaped_data[date] = []
            reshaped_data[date].append({
                "val": item["val"],
                # "accn": item["accn"],
                # "fy": item["fy"],
                # "fp": item["fp"],
                # "form": item["form"],
                "filed": item["filed"],
                # "frame": item.get("frame") 
            })
        gaps = []
    else:
        if annual:
        #We need the yearly values
            connected = []
            for datapoint in datapoint_list:
                if datapoint["form"] in ["8-K","10-K/A", "10-K", "20-F", "6-K"] and timediff(datapoint["start"], datapoint["end"])>340:
                    connected.append(datapoint)
        else:      
            #We need to get quartertly data for all of the quarters 
            #This should create data that is connected, whenever there is a link missing, we construct it
            #The data is sorted by the end date 
            #When two points are used to infer data the later filing date is asigned to the new point 
            for item in datapoint_list:  
                item['dur'] = (item['end'] - item['start']).days
            connected = []
            gaps = []
            wanted_end = datapoint_list[0]["end"]
            total_end = datapoint_list[-1]["end"]
            # The for loop keeps working until there is a hole, then the replace mechanism kicks in, after its done, the engine starts again from the start until it gets to where it was and goes again
            while(wanted_end < total_end): 
                missing = True
                for datapoint in datapoint_list:
                    if timediff(wanted_end,datapoint["end"]) < 10 and datapoint["dur"] < 100:
                        connected.append(datapoint)
                        wanted_end = datapoint["end"] + pd.Timedelta(days=91)
                        missing = False
                    elif (datapoint["end"] - wanted_end).days > 370:
                        break
                #missing 
                synthesised = False
                useful_ends = [wanted_end -pd.Timedelta(days=280),wanted_end + pd.Timedelta(days=370)]
                pieces = [piece for piece in datapoint_list if useful_ends[0]< piece["end"] <useful_ends[1]]
                # Find the interval of possible points to use to infer
                synthesised = False
                candidates = [] #We will get all the ways to get the thing here and pick the one with the best filed date
                for i,piece1 in enumerate(pieces,start=1):
                    for piece2 in pieces[i:]:
                        if abs(piece1["dur"] - piece2["dur"]) <100: #If the periods have a difference representing a quarter
                            if piece1["dur"] > piece2["dur"]: #Piece one is the longer duration 
                                if timediff(piece1["end"],piece2["end"]) <6: #If they match by their ends 
                                    if timediff(piece2["start"], wanted_end) <10: #Check if we are actually getting what we want
                                        candidates.append({"end": piece2["start"], "start":piece1["start"], "val": piece1["val"]-piece2["val"], "filed": max([piece1["filed"], piece2["filed"]])})

                                elif (timediff(piece1["start"], piece2["start"])) < 6: #If they match by their starts
                                    if timediff(piece1["end"], wanted_end) <10:
                                        candidates.append({"end": piece1["end"], "start": piece2["end"], "val": piece1["val"]-piece2["val"], "filed": max([piece1["filed"], piece2["filed"]])})

                            elif piece1["dur"] <piece2["dur"]:
                                if timediff(piece1["end"],piece2["end"]) <6: #If they match by their ends 
                                    if timediff(piece1["start"], wanted_end) <10:
                                        candidates.append({"end": piece1["start"], "start":piece2["start"], "val": piece2["val"]-piece1["val"], "filed": max([piece1["filed"], piece2["filed"]])})

                                elif (timediff(piece1["start"], piece2["start"])) < 6: #If they match by their starts
                                    if timediff(piece2["end"], wanted_end) <10:
                                        candidates.append({"end": piece2["end"], "start": piece1["end"], "val": piece2["val"]-piece1["val"], "filed": max([piece1["filed"], piece2["filed"]])})
                if candidates != []:
                    filed = candidates[0]["filed"]
                    index=0
                    for i, candidate in enumerate(candidates[1:],start=1):
                        if candidate["filed"] < filed:
                            filed = candidate["filed"]
                            index = i
                    diff = candidates[index]
                    diff["special"] = "synth_combo"
                    connected.append(diff)
                    synthesised = True
                    wanted_end = diff["end"] + pd.Timedelta(days=91)
                #This method introduces data that just is not true, thats why approx is important
                if not synthesised and approx:
                    shortest_duration = 1000
                    best_part = None
                    wanted_start = wanted_end - pd.Timedelta(days=91)
                    for piece in pieces:
                        if wanted_start- piece["start"] > -pd.Timedelta(days= 5) and  wanted_start - piece["end"] < pd.Timedelta(days= 5): # so that the interval is atleas kinda hugging the needed point
                            if shortest_duration > piece["dur"]:
                                best_part = piece
                                shortest_duration = piece["dur"]
                    if best_part != None:
                        # print(best_part)
                        synth_val = best_part["val"]* (91 / best_part["dur"]) #Reduce the value by the interval ratio
                        connected.append({"end": wanted_end, "start": wanted_start, "val": synth_val, "filed": best_part["filed"], "special": "synth_div"})
                        wanted_end = wanted_end + pd.Timedelta(days=91)
                        synthesised = True

                if synthesised == False and missing == True:
                    connected.append({"end": wanted_end, "start": wanted_end - pd.Timedelta(days=91), "val": np.nan, "filed": pd.Timestamp(year=1993, month=1, day=1), "special":"missing"})
                    gaps.append(wanted_end)
                    wanted_end = wanted_end + pd.Timedelta(days=91)
        #After getting the connected data, treat it the same as static so that you can use the same method and have consistency
        reshaped_data = {}
        for item in connected:
            date = item["end"]
            if date not in reshaped_data:
                reshaped_data[date] = []
            reshaped_data[date].append({
                "val": item["val"],
                "start": item["start"],
                # "accn": item["accn"],
                # "fy": item["fy"],
                # "fp": item["fp"],
                # "form": item["form"],
                "filed": item["filed"],
                "special": item.get("special")
                # "frame": item.get("frame") 
            })
    intervals = []
    reshaped = list(reshaped_data.items())
    interval_start = None
    i = 0
    while(i < len(reshaped)):
        key, value = reshaped[i]
        if interval_start == None:
            if not np.isnan(value[0]["val"]):
                interval_start = key
        elif np.isnan(value[0]["val"]):
            intervals.append((interval_start, reshaped[i-1][0]))
            interval_start = None
        i+=1
    if interval_start is not None:
        intervals.append((interval_start, reshaped[-1][0]))
    #Since we did not have it precomputed, we save it before we return it
    with open(f"C:\\Programming\\Python\\Finance\\EDGAR\\reshaped\\{ticker}\\{measure}.pkl", "wb") as file:
        pickle.dump((reshaped_data, intervals, dynamic), file)
    return reshaped_data, intervals, dynamic