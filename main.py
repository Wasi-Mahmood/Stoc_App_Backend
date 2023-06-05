from flask import Flask, jsonify, request
import pandas as pd
from pandas import DataFrame as df 
import json
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
CORS(app, origins=['http://localhost:3000'])  # Replace with the actual origin of your React app

#test_path_to_pred = "E:\Semesters\Fyp prepation\Stock-Market-Prediction-Using-LSTM-and-Online-News-Sentiment-Analysis\graph_data\MSFT\MSFT_time_y_test_prediction.csv"

def getDataset(ticker):
    try:
        # df =pd.read_csv(f"E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\{ticker}\{ticker}_test_pred_.csv")
        df =pd.read_csv(f"E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\{ticker}\{ticker}_predections_.csv")

        df = df.rename(columns={'pred+0': 'close'})
        df_to_dict = {row[0]: {'close' :format(row[1])} for row in df.values}
        dict_to_json = json.dumps(df_to_dict)
        return dict_to_json
    except:
        return json.dumps({'error': 404})


def getPrevPredDataset(ticker):
    try:
        # df =pd.read_csv(f"E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\{ticker}\{ticker}_test_pred_.csv")
        df =pd.read_csv(f"E:\Semesters\Fyp prepation\Forcasted_Data\Simple_Forcast\{ticker}\{ticker}_test_pred_.csv")
        df= df[-100:]
        df = df.rename(columns={'pred+0': 'close'})
        df_to_dict = {row[0]: {'close' :format(row[1])} for row in df.values}
        dict_to_json = json.dumps(df_to_dict)
        return dict_to_json
    except:
        return json.dumps({'error': 404})
    
    


@app.route('/pred', methods=['GET'])
def hello_world():
    ticker = request.args.get('ticker')
    return getDataset(ticker.upper())


@app.route('/PrevPred', methods=['GET'])
def getPrevPred():
    ticker = request.args.get('ticker')
    print(ticker)
    return getPrevPredDataset(ticker.upper())


if __name__ == '__main__':
    app.run(debug= True)