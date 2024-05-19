import httpx
import asyncio
from fredapi import Fred
import requests
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf

fred = Fred(api_key='0c34c4dd2fd6943f6549f1c990a8a0f0') 

async def fetch(url, url_headers, semaphore, client, timeout, max_retries, start_retry_delay):
        #function to fetch data from some url with retries and error responces
        for attempt in range(1,max_retries):
            try:
                async with semaphore:
                    response = await client.get(url, timeout=timeout, headers= url_headers)
                    response.raise_for_status()
                    return response  # Successful request, exit the loop
            except httpx.HTTPStatusError as e:
                    headers = response.headers
                    #Sometimes a retry-after header is returned
                    retry_after = headers.get('Retry-After')
                    if retry_after != None:
                        #Just for debugging
                        print(retry_after)
                        await asyncio.sleep(int(retry_after))
                        continue
                    if e.response.status_code == 404:
                        return "del"
                    print(f"Error response {e.response.status_code} for {url}")
            except httpx.TimeoutException as e:
                print(f"Timeout reached: {e}")
                print(f"Retrying in {attempt*start_retry_delay} seconds...")
                await asyncio.sleep(attempt*start_retry_delay)
            except httpx.RequestError as e:
                print(f"An error occurred: {e}.")
                await asyncio.sleep(attempt*start_retry_delay)
        return 0
                

def fred_info(ids:list, start:str, end:str):
    #Returns a dataframe with all of the indicators together
    #start and end are datatime objects
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    series_list = []
    for id in ids:
        series = fred.get_series(id,observation_start=start, observation_end=end)
        series_list.append(series)
    frame = pd.concat(series_list, axis=1, join="outer")
    frame.columns = ids
    return frame

async def yahoo_fetch(ticker, start_year, end_year, semaphore, max_retries, start_retry_delay):
    #Fetch implemented to get the price data for the company 
    async with semaphore:
        for attempt in range(max_retries +1):
            try:
                response = await asyncio.to_thread(si.get_data,ticker,min(start_year["static"], start_year["dynamic"]), max(end_year["static"], end_year["dynamic"]))
                return response  # Successful request, exit the loop
            except requests.exceptions.ConnectionError as ce:
                print("Yahoo connection error.")
                await asyncio.sleep(attempt*start_retry_delay)
            except Exception as e:
                print(f"Yahoo error:{e}")
                await asyncio.sleep(attempt*start_retry_delay)
        return 0
    
async def yahoo_info_fetch(ticker, max_retries, start_retry_delay):
    stock = yf.Ticker(ticker)
    info = 0
    for attempt in range(max_retries + 1):
        try:
            # Access the info property directly in a thread to avoid blocking the asyncio loop
            info = await asyncio.to_thread(lambda: stock.info)
            if info != None:
                break  
            await asyncio.sleep(attempt * start_retry_delay)
        except requests.exceptions.ConnectionError as ce:
            print(f"Share yahoo connection error: {ce} {ticker}")
            await asyncio.sleep(attempt * start_retry_delay)  # Implement exponential backoff
        except Exception as e:
            print(f"Share yahoo error: {e} {ticker}")
            return 0  
    return info

async def yahoo_split_fetch(ticker, max_retries, start_retry_delay):
    splits = 0
    for attempt in range(max_retries + 1):
        try:
            splits = await asyncio.to_thread(si.get_splits, ticker)
            break  
        except requests.exceptions.ConnectionError as ce:
            print(f"Split yahoo connection error: {ce} {ticker}")
            await asyncio.sleep(attempt * start_retry_delay)  # Implement exponential backoff
        except AssertionError:
            return pd.DataFrame() #Return something where there are no splits 
        except KeyError:
            return pd.DataFrame() #Return something where there are no splits
        except Exception as e:
            print(f"Split yahoo error: {e} {ticker}")
            return 0  
    return splits

async def yahoo_dividend_fetch(ticker, max_retries, start_retry_delay):
    splits = 0
    for attempt in range(max_retries + 1):
        try:
            splits = await asyncio.to_thread(si.get_dividends, ticker)
            break  
        except requests.exceptions.ConnectionError as ce:
            print(f"Dividend yahoo connection error: {ce} {ticker}")
            await asyncio.sleep(attempt * start_retry_delay)  # Implement exponential backoff
        # except AssertionError:
        #     return pd.DataFrame() #Return something where there are no splits 
        except Exception as e:
            print(f"Dividend yahoo error: {e} {ticker}")
            return 0  
    return splits


TIMEOUT = 8
RETRIES = 2
START_RETRY_DELAY = 3