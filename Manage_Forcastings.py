import requests
import pandas as pd
import os
from GetTimeSeriesPredictions import simpleTimeSeriesPred_LSTM
import datetime



def getStockData(ticker, timePeriod = "compact" "full",Alpha_vantage_API_KEY= "4KI6MT403YQOW06H" ):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={Alpha_vantage_API_KEY}&&outputsize={timePeriod}&datatype=json&interval=5min&time_period=60"
    return url


    
def json_to_dataframe(json, dataLength =0.8):
    df = pd.DataFrame.from_dict(json, orient='index')
    df.index.name = 'date'
    df = df.rename(columns={'date': 'date',"1. open": "open" ,"2. high":"high","3. low":"low",
                                        "4. close":"close","5. adjusted close":"adjusted close","6. volume":"volume",
                                        "7. dividend amount":"dividend amount","8. split coefficient":"split coefficient"})
    df = df.iloc[::-1]
    df.index =pd.to_datetime(df.index)



    percentage = dataLength
    indx = int(len(df) * percentage)
    df = df[indx:]
    return df

def save_df_to_csv(df, ticker,file_name,folder_path="E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\/"):
    # Check if the folder exists and create it if it doesn't
    if not os.path.exists(f"{folder_path}{ticker}"):
        os.makedirs(f"{folder_path}{ticker}")

    # Save the DataFrame to a CSV file in the specified folder
    df.to_csv(os.path.join(f"{folder_path}{ticker}", f"{ticker}_{file_name}.csv"))
    
    
    
    
    

def getSaveSimpleTimeSeriesPred_LSTM(ticker,dataframe =None, loop_back=7, number_of_dense_layers=2,timePeriod = "compact" "full", data_Length =0.79,save_results = False, folder_path="E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\/",special_deco_fileName=""):
    
    if dataframe == None:
        try:
            print("Fetching the stock Data from Alpha Vintage...")
            res = requests.get(getStockData(ticker,timePeriod))
            stockData = res.json()
            stockData= stockData["Time Series (Daily)"]
            dataframe = json_to_dataframe(stockData, data_Length)
            print("Fetching Sucessfull! ")
        except:
            print("Failed to Fetch Data from Alpha Vintage")
            raise     
        
        
    try:
        print("Getting predictions")
        predections, train_pred, test_pred, model = simpleTimeSeriesPred_LSTM(dataframe,loop_back, number_of_dense_layers)
        print("Got the Predictions Sucessfully!")
    except:
        print("Failed to Get Predictions")
        raise
    
    if save_results!= False:
        try:
            save_df_to_csv(predections,ticker,f"predections_{special_deco_fileName}", folder_path)
            save_df_to_csv(train_pred,ticker,f"train_pred_{special_deco_fileName}", folder_path)
            save_df_to_csv(test_pred,ticker,f"test_pred_{special_deco_fileName}", folder_path)
            print("Results Saved Sucesfully! ")
        except:
            print("Failed to Save the Results")
    
    return predections, train_pred, test_pred, model
