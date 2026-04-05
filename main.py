import func
import predict
import os
import parameter as para
import pandas as pd

# --------------------------------- Setup ----------------------------------
ship_type = 'CONTAINER' # make prediction and schedules for this ship type;|
train_start_time_line = "2023-01-01"
train_end_time_line = "2025-01-01"
sigma = 'sigma'
# -------------------------------- End ---------------------------------

# func.createFolder(para.figure_path)
# func.mergeDeparted()
data = func.read_data(ship_type)

training_data = data[(data[para.due_eta] >= train_start_time_line) & (data[para.due_eta] < train_end_time_line)]
test_data = data[(data[para.due_eta] >= train_end_time_line)]

from sklearn.preprocessing import OneHotEncoder
import numpy as np

unique_agent_names = test_data[para.in_port_agent_name].unique()
agent_sigma = {}
for agent_name in unique_agent_names:
    agent_data = training_data[training_data[para.in_port_agent_name] == agent_name]
    sample_z = agent_data[para.arrival_delay].values.astype(float)
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(agent_data[para.categorical_features])
    numeric_data = agent_data[para.numeric_features].values
    sample_xi = np.concatenate([numeric_data, encoded], axis=1).astype(float)
    if len(sample_z) > 0:
        alpha_list, beta, sigma = func.predict_and_estimate(sample_z, sample_xi)
        agent_sigma[agent_name] = sigma
test_data[sigma] = test_data[para.in_port_agent_name].map(agent_sigma)



results_month, results_test = predict.goRollingByMonth(data)
# results_month.to_csv("../Predict/predict-results-month.csv", index=False)
# results_test.to_csv("../Predict/predict-results-vessel.csv", index=False)
# print(results_month[['R2_train', 'R2_test', 'RMSE', 'MAE']].mean().round(2))
# print("=== Hava finished rolling prediction by month lasso regression ===")
#======================================================================================================
#===================================== Generate instances ========================================
results_test = pd.read_csv("../Predict/predict-results-vessel.csv")
results_test[para.due_eta] = pd.to_datetime(results_test[para.due_eta])
results_test[para.due_eta_day] = results_test[para.due_eta].dt.date
results_test[para.arrived_arrival_time] = pd.to_datetime(results_test[para.arrived_arrival_time])
results_test['predicted_arrival_time'] = results_test[para.due_eta] + pd.to_timedelta(results_test['pred'], unit='h')
#======================================================================================================
#===================================== Change datetime ========================================
func.createFolder(para.instance_path)

solved_days = os.listdir(para.instance_path)

for day, day_data in results_test.groupby(para.due_eta_day):
    csv_name = f'{day}.csv'
    if csv_name in solved_days: continue
    if (len(day_data) < 6): continue
    print(f'=== berth schedule on {day} ===')
    base_date = min(day_data[para.arrived_arrival_time].min(), day_data[para.due_eta].min(), day_data['predicted_arrival_time'].min())
    ddl = pd.to_datetime(day) + pd.Timedelta(hours=24)
    # T = func.twoDecimal((ddl - base_date).total_seconds() / 3600)
    
    day_data['ATA'] = func.twoDecimal((day_data[para.arrived_arrival_time] - base_date).dt.total_seconds() / 3600)
    day_data['ETA'] = func.twoDecimal((day_data[para.due_eta] - base_date).dt.total_seconds() / 3600)
    day_data['PTA'] = func.twoDecimal((day_data['predicted_arrival_time'] - base_date).dt.total_seconds() / 3600)
    day_data["s"] = func.twoDecimal(day_data[para.departed_service_time])
    delta_result = predict.getDelta(f"{day}.csv", "Month", "predict_error")
    asymmetric_delta_result = predict.getAsymmetricDelta(f"{day}.csv", "Month", "predict_error")
    day_data = predict.mergeDelta(day_data, delta_result, asymmetric_delta_result)
    day_data.to_csv(f'{para.instance_path}{day}.csv', index=False)
#======================================================================================================