#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 08:33:11 2025

@author: zhoubingjie
"""
import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import parameter as para
import figure
import rsome as rso
from rsome import ro
from rsome import cpt_solver as cpt
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from geopy.distance import geodesic
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from itertools import islice
import statsmodels.api as sm

# --- Utility Function ---
def oneDecimal(number):
    return(round(number, 1))

def twoDecimal(number):
    return(round(number, 2))

def insertColumn(df, column_name, new_col, new_data):
    if new_col in df.columns:
        df.drop(columns=[new_col], inplace=True)
    df.insert(df.columns.get_loc(column_name) + 1, new_col, new_data)

def createFolder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    
def getGroupCounts(data, classify_name):
    group_counts = (data.groupby(classify_name)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True))
    return group_counts
    
def getSubData(data, column_name, specific_value):
    return data[data[column_name] == specific_value]

def getCsvFile(group_counts):
    group_counts.to_csv("group_counts.csv", index=False, encoding="utf-8-sig")

def changeDatetime(data):
    data[para.arrived_arrival_day] = pd.to_datetime(data[para.arrived_arrival_day])
    data[para.arrived_arrival_time] = pd.to_datetime(data[para.arrived_arrival_time])
    data[para.due_eta] = pd.to_datetime(data[para.due_eta])
    data[para.due_eta_day] = data[para.due_eta].dt.date
    data[para.in_port_imo_no] = data[para.in_port_imo_no].astype(str)

def returnPastHourCol(pasthour, stat):
    return(f'past_{pasthour}hour_{stat}')

def returnSameHourCol(pastday, stat):
    return(f'past_{pastday}day_same_hour_{stat}')

def returnDayCol(pastday, stat):
    return(f'past_{pastday}day_{stat}')

def returnCurrentDayCol(stat):
    return(f'currentday_{stat}')

def extractZip():
    zip_set = [zip for zip in os.listdir(para.zip_path) if (zip != para.mac_file) and (zip.endswith(".zip"))]
    for zip in zip_set:
        xml_folder_name = os.path.splitext(os.path.basename(zip))[0]
        extract_folder = os.path.join(para.xml_path, xml_folder_name)
        createFolder(extract_folder)
        with zipfile.ZipFile(os.path.join(para.zip_path, zip), 'r') as zip_data:
            for member in zip_data.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue  
                source = zip_data.open(member)
                target_path = os.path.join(extract_folder, filename)
                with open(target_path, "wb") as target:
                    target.write(source.read())

def getCSV(start_name):
    xml_folders = [folder for folder in os.listdir(para.xml_path) if (folder != para.mac_file) and (folder.startswith(start_name))] #all XML folder names
    error_files = []
    xml_folder_data = pd.DataFrame()
    for xml_folder in xml_folders: 

        # print(f'xml_file_name is {xml_folder}')

        xml_files = [f for f in os.listdir(os.path.join(para.xml_path, xml_folder)) if f.endswith(".XML")]
        csv_folder = os.path.join(para.csv_path, xml_folder)
        createFolder(csv_folder)
        for xml_file in xml_files:

            # print(f'processing xml_file_name is {xml_file}')

            csv_file = xml_file.replace('.XML', '.csv')
            try:
                tree = ET.parse(os.path.join(para.xml_path, xml_folder, xml_file))
                root = tree.getroot()
                data = []
                for entry in root.findall('G_SQL1'):
                    row = {child.tag: child.text for child in entry}
                    data.append(row)
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(csv_folder, csv_file), index=False)
                xml_folder_data = pd.concat([xml_folder_data, df], ignore_index=True)
            except ET.ParseError as e:
                print(f"skip destroyed file: {xml_file}")
                error_files.append((xml_file, str(e)))
                continue
    xml_folder_data.drop_duplicates().to_csv(os.path.join(para.merge_path, para.data_type[start_name] + '.csv'), index=False)

def mergeDeparted():
    departed_data = pd.read_csv(os.path.join(para.merge_path, 'DEPARTED.csv')).rename(columns=lambda c: c.strip().upper()).drop_duplicates()
    departed_data = departed_data.sort_values(by='ATD_TIME')
    arrived_data = pd.read_csv(os.path.join(para.merge_path, para.merge_filename))
    arrived_data[para.arrived_arrival_time] = pd.to_datetime(arrived_data[para.arrived_arrival_time], errors='coerce')
    departed_data["ATD_TIME"] = pd.to_datetime(departed_data["ATD_TIME"], errors='coerce')
    merged_rows = []
    
    i = 0
    for idx, row in arrived_data.iterrows():
        i = i + 1
        print(f"--- here is merging {i}-th arrived data ---")
        nrow = {}
        merge_departed = False
        arrived_match = pd.DataFrame([row])
        mask = (
                ((departed_data["CALL_SIGN"] == row[para.arrived_call_sign]) | pd.isna(row[para.arrived_call_sign])) &
                ((departed_data["VESSEL_NAME"] == row["ARRIVED_VESSEL_NAME"]) | pd.isna(row["ARRIVED_VESSEL_NAME"])) 
            )
        match = departed_data[mask].copy()
        if match.empty:
            fmatch = match.copy()
        else:
            match[para.service_time] = (match["ATD_TIME"] - row[para.arrived_arrival_time]).dt.total_seconds() / 3600
            fmatch = match.loc[match[para.service_time] > 0].head(1)
        
        if not fmatch.empty:
            for col in arrived_match.columns:
                nrow[col] = arrived_match.iloc[0][col]
            for col in fmatch.columns:
                nrow[f'DEPARTED_{col}'] = fmatch.iloc[0][col]
                merge_departed = True
        if len(nrow) == 0 or not merge_departed: continue
        merged_rows.append(nrow)
    data = pd.DataFrame(merged_rows)
    data.to_csv(os.path.join(para.merge_path, 'mergeDeparted.csv'), index=False)
    
    
    
def mergeTime():
    
    def fillMissing(col_name, fill_list):
        data[col_name] = data[fill_list].bfill(axis=1).iloc[:, 0]
    
    data_name_set = ['ARRIVED', 'DUE', "IN_PORT"]
    # data_name_set = ['ARRIVED', 'DUE', "IN_PORT", "DEPARTED"]
    file_paths = {file:os.path.join(para.merge_path, file + '.csv') for file in data_name_set}
    dataframes = {
        name: pd.read_csv(path).rename(columns=lambda c: c.strip().upper()).drop_duplicates()
        for name, path in file_paths.items()
    }
    arrived_data = dataframes["ARRIVED"]
    due_data = dataframes["DUE"]
    inport_data = dataframes["IN_PORT"]
    # departed_data = dataframes["DEPARTED"]
    
    due_data["ETA"] = pd.to_datetime(due_data["ETA"], format="%Y/%m/%d %H:%M", errors='coerce')
    arrived_data["ARRIVAL_TIME"] = pd.to_datetime(arrived_data["ARRIVAL_TIME"], errors='coerce')
    # departed_data["ATD_TIME"] = pd.to_datetime(departed_data["ATD_TIME"], errors='coerce')
    inport_data["ARRIVAL_TIME"] = pd.to_datetime(inport_data["ARRIVAL_TIME"], format="%Y/%m/%d %H:%M", errors='coerce')
    # arrived_data["ARRIVAL_TIME"] = pd.to_datetime(arrived_data["ARRIVAL_TIME"], format="%d/%m/%Y %H:%M", errors='coerce')
    
    arrived_data = dataframes['ARRIVED'].drop(columns="REMARK").drop_duplicates()
    arrived_data = arrived_data.drop(columns=["CURRENT_LOCATION"]).drop_duplicates()
    merged_rows = []
    
    i = 0
    for idx, row in arrived_data.iterrows():
        i = i + 1
        print(f"--- here is merging {i}-th arrived data ---")
        
        # idx, row = next(arrived_data.iterrows())
        nrow = {}
        merge_due = False
        merge_inport = False
        merge_departed = True
        for name, df in dataframes.items():
            # name, df = next(itertools.islice(dataframes.items(), 3, 4))
            if (name == "ARRIVED"): continue
            arrived_match = pd.DataFrame([row])
            if (name == "DEPARTED"):
                mask = (
                        ((df["AGENT_NAME"] == row["AGENT_NAME"]) | pd.isna(row["AGENT_NAME"])) &
                        ((df["CALL_SIGN"] == row["CALL_SIGN"]) | pd.isna(row["CALL_SIGN"])) &
                        ((df["VESSEL_NAME"] == row["VESSEL_NAME"]) | pd.isna(row["VESSEL_NAME"])) 
                    )
            else:
                mask = (
                        ((df["SHIP_TYPE"] == row["SHIP_TYPE"]) | pd.isna(row["SHIP_TYPE"])) &
                        ((df["AGENT_NAME"] == row["AGENT_NAME"]) | pd.isna(row["AGENT_NAME"])) &
                        ((df["CALL_SIGN"] == row["CALL_SIGN"]) | pd.isna(row["CALL_SIGN"])) &
                        ((df["VESSEL_NAME"] == row["VESSEL_NAME"]) | pd.isna(row["VESSEL_NAME"])) 
                    )
            
            match = df[mask].copy()
            if (name == "DUE"):
                match[para.arrival_delay] = (row["ARRIVAL_TIME"]-match["ETA"]).dt.total_seconds() / 3600
                match = match[(match[para.arrival_delay] >= -para.delay_deviation) & (match[para.arrival_delay] <= para.delay_deviation)]
            
            if match.empty:
                fmatch = match.copy()
            else:
                if name == "DUE":
                    fmatch = match.loc[[match[para.arrival_delay].idxmin()]]
                elif (name == "IN_PORT"):
                    time_diff = (match["ARRIVAL_TIME"] - row["ARRIVAL_TIME"]).abs()
                    fmatch = match.loc[time_diff <= pd.Timedelta(minutes=5)].head(1)
                elif (name == "DEPARTED"):
                    time_diff = (match["ATD_TIME"] - row["ARRIVAL_TIME"]).abs()
                    fmatch = match.loc[time_diff > pd.Timedelta(minutes=5)].head(1)
            
            if not fmatch.empty:
                for col in arrived_match.columns:
                    nrow[f'ARRIVED_{col}'] = arrived_match.iloc[0][col]
                for col in fmatch.columns:
                    if col==para.arrival_delay:
                        nrow[col] = fmatch.iloc[0][col]
                    nrow[f'{name}_{col}'] = fmatch.iloc[0][col]
                    if name == "DUE":
                        merge_due = True
                    elif name == "IN_PORT":
                        merge_inport = True
                    elif name == "DEPARTED":
                        merge_departed = True
        if len(nrow) == 0 or not merge_due or not merge_inport or not merge_departed: continue
        merged_rows.append(nrow)
    data = pd.DataFrame(merged_rows)
    data.to_csv(os.path.join(para.merge_path, para.raw_merge_filename), index=False)
    fillMissing(para.arrived_ship_type, [para.due_ship_type])
    fillMissing(para.arrived_agent_name, [para.due_agent_name])
    data = data[data[para.due_eta].dt.minute.isin([0, 30])] # delete wrong ETA
    data = data.dropna(subset=[para.in_port_imo_no])  # delete NaN
    data[para.in_port_imo_no] = data[para.in_port_imo_no].astype(str)
    data = data[~data[para.in_port_imo_no].astype(str).str.startswith("TEMP")]
    data = data[(data[para.due_pasi] == "RCVD")]
    addFeature(data)
    data.to_csv(os.path.join(para.merge_path, para.merge_filename), index=False)

#process data    
def prepareData():
    # createFolder(para.merge_path)
    # extractZip()
    # for start_name in para.data_type.keys():
    #     getCSV(start_name)
    # mergeTime()
    print('--- merge data successfully ---')

#self-defined features
def addFeature(data):
    data[para.arrived_arrival_time] = pd.to_datetime(data[para.arrived_arrival_time])
    data[para.arrived_arrival_month] = data[para.arrived_arrival_time].dt.month
    data[para.arrived_arrival_day] = data[para.arrived_arrival_time].dt.date
    data[para.arrived_arrival_hour] = data[para.arrived_arrival_time].dt.hour
    data[para.arrived_arrival_weekday] = data[para.arrived_arrival_time].dt.strftime('%A')
    
    for idx,row in data.iterrows():
        #{current_day_arrival_delay} is the delay on current day before current vessel
        past_data = data[(data[para.arrived_arrival_day] == row[para.arrived_arrival_day]) & (data[para.arrived_arrival_time] < row[para.arrived_arrival_time])]
        for stat in para.stats:
            data.loc[idx, returnCurrentDayCol(stat)] = 0 if len(past_data) == 0 else past_data[para.arrival_delay].agg(stat)
        #arrival delay before {pasthour} of the current arrival time
        for pasthour in para.pasthour_set:
            past_data = data[(data[para.arrived_arrival_time] >= row[para.arrived_arrival_time] - timedelta(hours=pasthour)) & (data[para.arrived_arrival_time] < row[para.arrived_arrival_time])]
            for stat in para.stats:
                data.loc[idx, returnPastHourCol(pasthour, stat)] = 0 if len(past_data) == 0 else past_data[para.arrival_delay].agg(stat)
    
    for day, current_data in data.groupby(para.arrived_arrival_day): #group by arrival day
        # day = date(2025, 6, 1)
        # current_data = data.groupby(para.arrived_arrival_day).get_group(day)
        for pastday in para.pastday_set: #add feature for each past day
            #total delay on the past day
            past_data = data[(data[para.arrived_arrival_day] == (day - timedelta(days=pastday)))] #records for the pstday-th day
            for stat in para.stats:
                data.loc[current_data.index, returnDayCol(pastday, stat)] = 0 if (len(past_data) == 0) else past_data[para.arrival_delay].agg(stat)
        #delay on the pastday in the same hour
        for hour in current_data[para.arrived_arrival_hour].unique(): #group by hour on current day further
            for pastday_same_hour in para.pastday_same_hour_set:
                past_data = data[(data[para.arrived_arrival_day] == day - timedelta(days=pastday_same_hour)) & (data[para.arrived_arrival_hour] == hour)]
                defined_mask = current_data[current_data[para.arrived_arrival_hour] == hour].index
                for stat in para.stats:
                    data.loc[defined_mask, returnSameHourCol(pastday_same_hour, stat)] = 0 if (len(past_data) == 0) else past_data[para.arrival_delay].agg(stat)
    data[para.shiptype_agentname] = data[para.arrived_ship_type].astype(str) + "_" + data[para.arrived_agent_name].astype(str)

def addVesselInformation(data):
    ship_infor = pd.read_csv("../Data/Web/shipInformation.csv")
    data[para.in_port_imo_no] = data[para.in_port_imo_no].astype(str)
    ship_infor["IMO"] = ship_infor["IMO"].astype("Int64").astype(str)

    data = data.merge(ship_infor[[para.imo, para.length, para.beam, para.GT, para.year]],
        left_on=para.in_port_imo_no,   
        right_on="IMO",             
        how="left"
    ).drop(columns=["IMO"], errors="ignore")
    return data

def getPortStat(data):
    port_stats = data.groupby(para.due_last_port)[para.arrival_delay].agg(
        delay_mean="mean",
        delay_std="std",
        delay_median="median",
        delay_count="count"
    ).reset_index().fillna(0)
    port_stats = port_stats.rename(columns={
        "delay_mean": para.arrival_delay_mean,
        "delay_std": para.arrival_delay_std,
        "delay_median": para.arrival_delay_median,
        "delay_count": para.arrival_delay_count
    })
    return port_stats

def addPortInformation(training_data, test_data):
    port_stats = getPortStat(training_data)
    training_data = training_data.merge(port_stats, on=para.due_last_port, how="left")
    test_data = test_data.merge(port_stats, on=para.due_last_port, how="left")
    return training_data, test_data

def printSummary(model, x_train, y_train, x_test, y_test, transformed_feature_names):
    
    # for feature_name, coef in zip(transformed_feature_names, model.coef_):
    #     if (coef != 0):
    #         print(f'Feature：{feature_name}，Coef：{coef}')    
    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)
    # print(f'r2 on training data: {r2_train}')
    # print(f'r2 on test data: {r2_test}')
    return r2_train, r2_test 

def getResidualMetric(err, quant=0.90):
    #err: ATA - ETA 
    e = np.asarray(err, dtype=float).ravel()
    mask = np.isfinite(e)
    e = e[mask]
    n = e.size
    out = dict(n=int(n))

    if n == 0:
        base = dict(bias_h=np.nan, medae_h=np.nan, rmse_h=np.nan,
                    p90ae_h=np.nan, early_rate=np.nan, mad_h=np.nan)
        out.update(base)
        return out

    ae = np.abs(e)
    out["bias"]   = float(np.mean(e))
    out["medae"]  = float(np.median(ae))
    out["rmse"]   = float(np.sqrt(np.mean(e**2)))
    out["p90ae"]  = float(np.quantile(ae, quant))
    out["early_rate"] = float(np.mean(e < 0))
    med = np.median(e)
    out["mad_h"]    = float(1.4826 * np.median(np.abs(e - med)))

    return out


def fitModel(x_train_raw, x_test_raw, y_train, y_test):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    category_encoded_train = encoder.fit_transform(x_train_raw[para.categorical_features])  
    category_encoded_test = encoder.transform(x_test_raw[para.categorical_features])  
    transformed_category_columns = encoder.get_feature_names_out(para.categorical_features)
    feature_names = np.concatenate((para.numeric_features, transformed_category_columns)) #

    scaler = StandardScaler() # standardization
    x_scaled_train = scaler.fit_transform(x_train_raw[para.numeric_features].values)  
    x_scaled_test = scaler.transform(x_test_raw[para.numeric_features].values)  

    x_train = np.column_stack([x_scaled_train, category_encoded_train])
    x_test = np.column_stack([x_scaled_test, category_encoded_test])
    
    # model = Lasso(alpha=para.alpha).fit(x_train, y_train)
    model = LassoCV(alphas=np.logspace(-4, 1, 50),  # 搜索范围 (10^-4 到 10^2，共50个值)
        cv=5,   
        max_iter=3000,   
        random_state=42).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2_train, r2_test = printSummary(model, x_train, y_train, x_test, y_test, feature_names)
    return y_pred, r2_train, r2_test

def fitModelQR(x_train_raw, x_test_raw, y_train, y_test, quantile):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    category_encoded_train = encoder.fit_transform(x_train_raw[para.categorical_features])  
    category_encoded_test = encoder.transform(x_test_raw[para.categorical_features])  
    transformed_category_columns = encoder.get_feature_names_out(para.categorical_features)

    scaler = StandardScaler() # standardization
    x_scaled_train = scaler.fit_transform(x_train_raw[para.numeric_features].values)  
    x_scaled_test = scaler.transform(x_test_raw[para.numeric_features].values)  

    x_train = np.column_stack([x_scaled_train, category_encoded_train])
    x_test = np.column_stack([x_scaled_test, category_encoded_test])
    
    x_train_const = sm.add_constant(x_train)   # 加常数项
    x_test_const = sm.add_constant(x_test)

    model = sm.QuantReg(y_train, x_train_const)
    res = model.fit(q=quantile)

    # ---------- 5. 预测 ----------
    y_pred_train = res.predict(x_train_const)
    y_pred_test = res.predict(x_test_const)
    
    r2_train = oneDecimal(1 - ((y_train - y_pred_train)**2).sum() / ((y_train - y_train.mean())**2).sum())
    r2_test = oneDecimal(1 - ((y_test - y_pred_test)**2).sum() / ((y_test - y_test.mean())**2).sum())

    print(res.summary())
    return y_pred_test, r2_train, r2_test

def goRandom(data, test_size=0.2, random_state=42):
    def calculate_metrics(y_true, y_pred, mae_label, rmse_label):
        return {
            mae_label: round(mean_absolute_error(y_true, y_pred), 2),
            rmse_label: round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        }
    
    
    
    X = data[para.raw_features]
    y = data[para.arrival_delay]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
        data, data, test_size=test_size, random_state=random_state
    )
    
    pred, r2_train, r2_test = fitModel(X_train, X_test, y_train, y_test)
    # pred, r2_train, r2_test = fitModelQR(X_train, X_test, y_train, y_test, quantile = 0.9)
    
    metrics_model = calculate_metrics(y_test, pred, 'MAE', 'RMSE')
    summary_model = pd.DataFrame([{'R2_train': round(r2_train, 2), 'R2_test': round(r2_test, 2), **metrics_model}])
    
    summary_prediction = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': pred,
        'Gap(hour)': y_test.values - pred
    })
    
    # insertColumn(data_y_test, para.arrival_delay, 'pred', pred)
    # insertColumn(data_y_test, 'pred', 'Gap', summary_prediction['Gap(hour)'].values)
    
    # summary_prediction['Gap(hour)'].describe()
    # debug_left = data_y_test[data_y_test['Gap'] < -10]
    # debug_right = data_y_test[data_y_test['Gap'] > 10]
    # gg = getGroupCounts(debug_right, para.length)
    # figure.plotTwoHist(debug_left, debug_right, para.length)
    # figure.plotTwoHist(debug_left, debug_right, para.GT)
    
    metrics_model = calculate_metrics(summary_prediction["y_true"], summary_prediction["y_pred"], 'MAE', 'RMSE')
    figure.plotRandomGap(summary_prediction, summary_model, "random")
    return summary_model

def calculate_metrics(y_true, y_pred, mae_label, rmse_label):
    return {
        mae_label: mean_absolute_error(y_true, y_pred),
        rmse_label: np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
def returnMetrics(y_test, pred, MAE, RMSE):
    return (calculate_metrics(y_test, pred, MAE, RMSE))

def goRollingByDay(data):
    results_list = []
    for day, day_data in data.groupby(para.arrived_arrival_day):
        # day, day_data = next(islice(data.groupby(para.arrived_arrival_day), 1, 1 + 1))
        print(f"=== here is predicting {day} ===")
        training_data = data[(data[para.arrived_arrival_day] >= day - timedelta(days=para.train_window_day)) & (data[para.arrived_arrival_day] < day)]
        if training_data.empty: continue
        training_day = len(training_data[para.arrived_arrival_day].unique())    
        if training_day < para.train_window_day: continue
        test_data = data[(data[para.arrived_arrival_day] >= day) & (data[para.arrived_arrival_day] <= day + timedelta(days=para.test_window_day))]
        if test_data.empty: continue
        test_day = len(test_data[para.arrived_arrival_day].unique())
        if test_day < para.test_window_day: continue 
        x_train_raw = training_data[para.raw_features]
        x_test_raw = test_data[para.raw_features]
        y_train = training_data[para.arrival_delay]
        y_test = test_data[para.arrival_delay]
        pred, r2_train, r2_test = fitModel(x_train_raw, x_test_raw, y_train, y_test)
        metrics_model = returnMetrics(y_test, pred, 'MAE', 'RMSE')
        results_list.append({'date': day, 'R2_train': r2_train, 'R2_test': r2_test, **metrics_model})
    return(pd.DataFrame(results_list))

def goRollingByMonth(data):
    
    def getTrainingData(training_data, test_data, test_month):
        test_month = str(test_month)
        train_store_path = "../Data/Training/"
        test_store_path = "../Data/Test/"
        createFolder(train_store_path)
        createFolder(test_store_path)
        training_data.to_csv(f"{train_store_path}{test_month}.csv", index=False)
        test_data.to_csv(f"{test_store_path}{test_month}.csv", index=False)
    
    year_month = "year_month"
    data[year_month] = data[para.due_eta].dt.to_period("M")
    results_list = []
    result_test_data = pd.DataFrame()
    all_months = sorted(data[year_month].unique())
    for i in range(12, len(all_months)):
        train_months = all_months[(i - 12):i]   
        test_month = all_months[i]
        print(f'=== Here is predict {i}/{len(all_months)}-th month ===')
        training_data = data[data[year_month].isin(train_months)]
        if training_data.empty: continue
        test_data = data[data[year_month] == test_month]
        if test_data.empty: continue
        
        getTrainingData(training_data, test_data, test_month) #analyze prediction deviations of vessel   
    
        x_train_raw = training_data[para.raw_features]
        x_test_raw = test_data[para.raw_features]
        y_train = training_data[para.arrival_delay]
        y_test = test_data[para.arrival_delay]
        pred, r2_train, r2_test = fitModel(x_train_raw, x_test_raw, y_train, y_test)
        summary_prediction = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': pred,
            'Gap(hour)': y_test.values - pred
        })
        insertColumn(test_data, para.arrival_delay, 'pred', pred)
        insertColumn(test_data, 'pred', 'Gap', summary_prediction['Gap(hour)'].values)
        
        metrics_model = returnMetrics(y_test, pred, 'MAE', 'RMSE')
        results_list.append({
            'train_period': f"{train_months[0]} to {train_months[-1]}",
            'test_period': str(test_month),
            'train_size': len(training_data),
            'test_size': len(test_data),
            'R2_train': r2_train,
            'R2_test': r2_test,
            **metrics_model
        })
        selected_columns = [para.arrived_call_sign, para.arrived_ship_type, para.arrived_agent_name, para.due_last_port, para.arrived_arrival_time, para.due_eta, para.arrival_delay, para.departed_service_time, 'pred']
        result_test_data = pd.concat([test_data[selected_columns], result_test_data], axis=0, ignore_index=True)
    return pd.DataFrame(results_list), result_test_data

def goPred(data, label_shiptype, label_figure):
    data['y'] = pd.to_datetime(data[para.arrived_arrival_time]).astype("int64") / 1e9
    data['x'] = pd.to_datetime(data[para.due_eta]).astype("int64") / 1e9
    
    X = data[['x']]   
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred_second = model.predict(X_test)  
    y_test_pred_time = pd.to_datetime(y_test_pred_second * 1e9)
    r2 = r2_score(y_test, y_test_pred_second)
    summary_model = pd.DataFrame({"R2": r2, "Coef": round(model.coef_[0], 2), "Intercept(h)": round(model.intercept_/3600, 2), "Count": len(data)}, index=[0])
    summary_prediction = pd.DataFrame({
        "ActualArrivalTime": pd.to_datetime(y_test * 1e9),
        "PredictedArrivalTime": y_test_pred_time
    })
    summary_prediction["Gap(hour)"] = (summary_prediction["ActualArrivalTime"] - summary_prediction["PredictedArrivalTime"]).dt.total_seconds() / 3600
    figure.plotGap(summary_prediction, summary_model, label_shiptype, label_figure)
    return summary_prediction

def goPredGroup(data, classify_name):
    stat_shiptype = (
        data.groupby(classify_name)
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)   
            .reset_index(drop=True)   
    ) #summarize the number of ship type
    shiptype_dict = dict(zip(stat_shiptype[classify_name], stat_shiptype["count"]))

    i = 1
    for ship_type in shiptype_dict.keys():
        data_type = data[data[classify_name] == ship_type]
        goPred(data_type, label_shiptype=ship_type, label_figure=f"{i}_{classify_name}")
        i = i + 1

#====== MVCS ======
def mvcs(z, s=None, r=None, p1=2, p2=2, display=True):
    
    N, L = z.shape #L is the dimension of z, N is the number of in-sample
    
    model = ro.Model()
    q = model.dvar(L)
    u = model.dvar(N)
    v = model.dvar(N)
    Q = model.dvar((L, L))
    
    if s is None:
        side = np.zeros(N)
    else:
        S = s.shape[1] #dimension of vector s
        P = model.dvar((L, S))
        side = s@P.T
    model.max(rso.rootdet(Q))
    model.st((1/N) * u.sum() <= 1)
    if isinstance(p2, tuple): 
        model.st((1/L) * rso.power(v, *p2) <= u) #p2 is a tuple, each element of p2 as an input parameter
    else:
        model.st((1/L) * rso.power(v, p2) <= u) #rso.power(v,p2) = v^p2
    l_norm = lambda x: rso.norm(x, p1)  #define a function 'l_norm': p-norm of x
    for n in range(N):
        model.st(l_norm(Q@z[n] - q - side[n]) <= v[n])
    
    if r is not None:
        model.st(Q << np.diag(r ** (-1)))  # regularization constraint            
    if display:
        msg = f"Sample data:       {N} records x {L} inputs \n"
        msg += "Side information:  "
        msg += "None\n" if s is None else f"{S} features\n"
        msg += f"Norm type:         l{p1}-norm\n"
        msg += f"Deviation penalty: power={p2}\n"
        print(msg)
        print(model.do_math())
    model.solve(cpt, display=display)
    if s is None:
        return q.get(), Q.get()                             # outputs with no side information
    else:
        return q.get(), P.get(), Q.get()                    # outputs with inside information
    
def goSideInforModel():
    m = 20
    L = 2
    train = pd.DataFrame(np.linspace(0, 2, m), columns=['x'])
    train['z1'] = train['x'] ** 2
    train['z2'] = 1.5 * train['x'] ** 3 - 1
    zs = train[['z1', 'z2']].values
    ss = train[['x']].values # side information
    q, P, Q = mvcs(zs, ss, p1=2, p2=2, display=False)
    D = np.linalg.inv(Q)
    Zhat = D @ P
    zhat0 = D @ q
    figure.plotContour(zs, ss, train, zhat0, Zhat, Q)
    
def writeExcelSheet(raw_data):
    with pd.ExcelWriter("../Data/GroupAISData.xlsx", engine='openpyxl') as writer:
        for mmsi, group in raw_data.groupby(para.mmsi):
            group.to_excel(writer, sheet_name=str(mmsi), index=False)
    
def compute_eta(row, port_center):
    try:
        distance_nm = geodesic((row[para.lon], row[para.lon]), port_center).nautical
        sog = row[para.sog]
        eta_hours = distance_nm / sog
        return row[para.date_time] + timedelta(hours=eta_hours)
    except:
        return pd.NaT    

def read_data(ship_type):
    raw_data = pd.read_csv(os.path.join(para.merge_path, para.merge_departed_filename)) # without any filter including all ship types
    data = raw_data[raw_data[para.arrived_ship_type] == ship_type].copy()
    changeDatetime(data) 
    data = addVesselInformation(data)
    data = data.dropna(subset=para.raw_features)
    data = data.sort_values(by=para.due_eta)
    data = data[(data[para.departed_service_time] >= 1) & (data[para.departed_service_time] <= 36)]
    # data[para.departed_service_time].describe()
    return data
