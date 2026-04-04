#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:54:11 2025

@author: zhoubingjie
"""
import parameter as para
import os
import pandas as pd
import numpy as np
from rsome import ro
import rsome as rso
from rsome import cpt_solver as cpt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV

def insertColumn(df, column_name, new_col, new_data):
    if new_col in df.columns:
        df.drop(columns=[new_col], inplace=True)
    df.insert(df.columns.get_loc(column_name) + 1, new_col, new_data)

def calculate_metrics(y_true, y_pred, mae_label, rmse_label):
    return {
        mae_label: mean_absolute_error(y_true, y_pred),
        rmse_label: np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
def returnMetrics(y_test, pred, MAE, RMSE):
    return (calculate_metrics(y_test, pred, MAE, RMSE))

def getTrainingData(training_data, test_data, test_month):
    test_month = str(test_month)
    train_store_path = "../Data/Training/"
    test_store_path = "../Data/Test/"
    os.makedirs(train_store_path, exist_ok=True)
    os.makedirs(test_store_path, exist_ok=True)
    training_data.to_csv(f"{train_store_path}{test_month}.csv", index=False)
    test_data.to_csv(f"{test_store_path}{test_month}.csv", index=False)

def getActiveFeatures(x_train_raw):
    numeric_features = [feature for feature in para.numeric_features if feature in x_train_raw.columns]
    categorical_features = [
        feature for feature in para.categorical_features
        if feature in x_train_raw.columns and x_train_raw[feature].nunique(dropna=False) > 1
    ]
    return numeric_features, categorical_features

def fitModel(x_train_raw, x_test_raw, y_train, y_test):
    numeric_features, categorical_features = getActiveFeatures(x_train_raw)

    if categorical_features:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        category_encoded_train = encoder.fit_transform(x_train_raw[categorical_features])
        category_encoded_test = encoder.transform(x_test_raw[categorical_features])
        transformed_category_columns = encoder.get_feature_names_out(categorical_features)
    else:
        category_encoded_train = np.empty((len(x_train_raw), 0))
        category_encoded_test = np.empty((len(x_test_raw), 0))
        transformed_category_columns = np.array([], dtype=object)

    feature_names = np.concatenate((numeric_features, transformed_category_columns))

    scaler = StandardScaler() # standardization
    x_scaled_train = scaler.fit_transform(x_train_raw[numeric_features].values)
    x_scaled_test = scaler.transform(x_test_raw[numeric_features].values)

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

def printSummary(model, x_train, y_train, x_test, y_test, transformed_feature_names):
    
    # for feature_name, coef in zip(transformed_feature_names, model.coef_):
    #     if (coef != 0):
    #         print(f'Feature：{feature_name}，Coef：{coef}')    
    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)
    # print(f'r2 on training data: {r2_train}')
    # print(f'r2 on test data: {r2_test}')
    return r2_train, r2_test 

def goRollingByMonth(data):
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

def getDelta(instance_name, training_data_folder, predict_error):
    # classify_name = para.due_last_port
    train_data_name = instance_name.replace(".csv", "").rsplit("-", 1)[0]
    train_data = pd.read_csv(f"../Data/{training_data_folder}/{train_data_name}.csv")
    mvcs_result = {}
    for name, group in train_data.groupby(para.due_last_port):
        z = abs(group[predict_error]).values.reshape(-1, 1)
        if z.shape[0] == 1: 
            mvcs_result[name] = z[0,0]
        else:
            mvcs_result[name] = mvcs(z)
    return mvcs_result

def getAsymmetricDelta(instance_name, training_data_folder, predict_error):
    # classify_name = para.due_last_port
    train_data_name = instance_name.replace(".csv", "").rsplit("-", 1)[0]
    train_data = pd.read_csv(f"../Data/{training_data_folder}/{train_data_name}.csv")
    mvcs_result = {}
    for name, group in train_data.groupby(para.due_last_port):
        z = abs(group[predict_error]).values.reshape(-1, 1)
        if z.shape[0] == 1: 
            mvcs_result[name] = z[0,0]
        else:
            mvcs_result[name] = mvcs_asymmetric(z)
    return mvcs_result

def mergeDelta(ins, delta_result, asymmetric_delta_result):
    ins["delta"] = ins[para.due_last_port].map(delta_result).fillna(0)
    ins["adelta"] = ins[para.due_last_port].map(asymmetric_delta_result).fillna(0)
    return ins

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
        
    if L == 1:
        model.max(Q[0, 0]) #Q is a scalar
    else:
        model.max(rso.rootdet(Q))
        
    # model.max(rso.rootdet(Q))
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
        Q_inv = np.linalg.inv(Q.get())
        return (Q_inv @ q.get())[0]                             # outputs with no side information
    else:
        return q.get(), P.get(), Q.get()                    # outputs with inside information

def mvcs_asymmetric(z, s=None, r=None, p1=2, p2=2, max_iter=1000, tol=2e-3, display=True):
    
    N, L = z.shape
    sigma1, sigma2 = np.ones(L), np.ones(L)
    volumn_before = np.inf
    for i in range(max_iter):
        if display:
            print(f'Iteration {i+1}:')
        
        if s is None:
            q, Q = qq_prob(z, sigma1, sigma2, p1=p1, p2=p2, display=display)
            side = None
        else:
            q, P, Q = qq_prob(z, sigma1, sigma2, s, p1=p1, p2=p2, display=display)
            side = s @ P.T

        v1, v2 = sigma_prob(z, Q, q, side, p1=p1, p2=p2, display=display)
        sigma1, sigma2 = 1/v1, 1/v2
        
        volumn_after = 1/np.linalg.det(Q) * np.prod((sigma1 + sigma2)/2)
        if display:
            print(f'The current volumn: {volumn_after}')
            print('')
        gap = abs(volumn_after - volumn_before) / volumn_after
        if gap < tol:
            break
        else:
            volumn_before = volumn_after
    if s is None:
        Q_inv = np.linalg.inv(Q)
        return (Q_inv @ q)[0]                             # outputs with no side information
    else:
        return q.get(), P.get(), Q.get()    

def qq_prob(z, sigma1, sigma2, s=None, r=None, p1=2, p2=2, display=True):
    
    N, L = z.shape
    
    model = ro.Model()
    q = model.dvar(L)
    h = model.dvar(N)
    g = model.dvar(N)
    u = model.dvar((2, N, L))
    Q = model.dvar((L, L))
    
    if s is None:
        side = np.zeros(N)
    else:
        S = s.shape[1]
        P = model.dvar((L, S))
        side = s@P.T
    
    if L == 1:
        model.max(Q[0, 0]) #Q is a scalar
    else:
        model.max(rso.rootdet(Q))

    model.st((1/N) * h.sum() <= 1)
    model.st((1/L) * rso.power(g, p2) <= h)
    l_norm = lambda x: rso.norm(x, p1)                      # p-norm of x
    for n in range(N):
        model.st(u[0, n] >= np.diag(1/sigma1)@(Q@z[n] - q - side[n]))
        model.st(u[1, n] >= -np.diag(1/sigma2)@(Q@z[n] - q - side[n]))
        model.st(l_norm(u[0, n] + u[1, n]) <= g[n])
    model.st(u >= 0)
    
    if r is not None:
        model.st(Q << np.diag(r ** (-1)))                   # regularization constraint    
        
    model.solve(cpt, display=display)
    
    if s is None:
        return q.get(), Q.get()
    else:
        return q.get(), P.get(), Q.get()

def sigma_prob(z, Q, q, side=None, p1=2, p2=2, display=True):
    
    N, L = z.shape
    
    model = ro.Model()
    t = model.dvar(L)
    v1 = model.dvar(L)
    v2 = model.dvar(L)
    w1 = model.dvar(L)
    w2 = model.dvar(L)
    h = model.dvar(N)
    g = model.dvar(N)
    u = model.dvar((2, N, L))
    
    if side is None:
        side = np.zeros(N)
    
    model.max(rso.gmean(t))
    model.st((1/N) * h.sum() <= 1)
    model.st((1/L) * rso.power(g, p2) <= h)
    l_norm = lambda x: rso.norm(x, p1)                      # p-norm of x
    for n in range(N):
        model.st(u[0, n] >= v1* (Q@z[n] - q - side[n]))
        model.st(u[1, n] >= -v2 * (Q@z[n] - q - side[n]))
        model.st(l_norm(u[0, n] + u[1, n]) <= g[n])
    model.st(u >= 0)
    for l in range(L):
        model.st(rso.rsocone(t[l], 2*v1[l], w1[l]))
        model.st(rso.rsocone(t[l], 2*v2[l], w2[l]))
    model.st(t >= w1 + w2)
    model.st(v1 >= 0, v2 >= 0, w1 >= 0, w2 >= 0)
    
    model.solve(cpt, display=display)
    
    return v1.get(), v2.get()
