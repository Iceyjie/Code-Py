#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 10:42:45 2025

@author: zhoubingjie
"""
mmsi = 'MMSI'
imo = "IMO"
length = "Length"
beam = "Beam" 
GT = "GT"
year = "Year"

#column names
mac_file = '.DS_Store'
vessel_name = 'vessel_name'.upper()
arrival_delay = 'arrival_delay'.upper()
due_eta = 'due_eta'.upper()

arrived_arrival_time = 'arrived_arrival_time'.upper()
in_port_arrival_time = 'in_port_arrival_time'.upper()

arrived_current_location = 'arrived_current_location'.upper()
in_port_current_location = 'in_port_current_location'.upper()

in_port_ship_type = 'in_port_ship_type'.upper()
arrived_ship_type = 'arrived_ship_type'.upper()
due_ship_type = 'due_ship_type'.upper()
departed_ship_type = 'departed_ship_type'.upper()

arrived_agent_name = 'arrived_agent_name'.upper()
in_port_agent_name = 'in_port_agent_name'.upper()
due_agent_name = 'due_agent_name'.upper()
departed_agent_name = 'departed_agent_name'.upper()

arrived_call_sign = 'arrived_call_sign'.upper()
in_port_call_sign = 'in_port_call_sign'.upper()
due_call_sign = 'due_call_sign'.upper()
departed_call_sign = 'departed_call_sign'.upper()
departed_atd_time = 'departed_atd_time'.upper()

in_port_imo_no = 'in_port_imo_no'.upper()
in_port_flag = 'in_port_flag'.upper()

due_pasi = 'due_pasi'.upper()

#added features
arrived_arrival_month = 'arrived_arrival_month'
arrived_arrival_day = 'arrived_arrival_day'
arrived_arrival_hour = 'arrived_arrival_hour'
arrived_arrival_weekday = 'arrived_arrival_weekday'
departed_last_berth = 'departed_last_berth'.upper()
due_last_port = 'due_last_port'.upper()
shiptype_agentname = 'shiptype_agentname'.upper()
past_arrival_mean = 'past_arrival_mean'.upper()
shiptype_lastport = 'shiptype_lastport'.upper()
arrival_delay_mean = 'arrival_delay_mean'.upper()
arrival_delay_median = 'arrival_delay_median'.upper()
arrival_delay_std = 'arrival_delay_std'.upper()
arrival_delay_count = 'arrival_delay_count'.upper()

due_eta_day = 'due_eta_day'.upper()
departed_service_time = 'departed_service_time'.upper()

#store file paths
# root_path = '/Users/zhoubingjie/Research/new'
root_path = "../Data/Merge"
xml_path = f'{root_path}/XML/'
csv_path = f'{root_path}/CSV/'
zip_path = f'{root_path}/ZIP/'

merge_path = '../Data/MERGE/'
figure_path = '../Figure/'
result_path = '../Result/'
instance_path = '../Data/Instance/'
data_path = '../Data/data/'


merge_filename = 'mergeResult.csv'
raw_merge_filename = 'rawMergeResult.csv'
merge_departed_filename = 'mergeDeparted.csv'

data_type = {'c9a': 'ARRIVED', '7cd': 'DUE', '689': 'IN_PORT', '024': 'DEPARTED'}
stats = ['mean', 'median']



#seld-defined parameters
delay_deviation = 24 #Hour
pastday_set = [i for i in range(1, 6)]
pastday_same_hour_set = [i for i in range(1, 6)]
pasthour_set = [i for i in range(1, 3)]
train_window_day = 52*7*2
test_window_day = 7
alpha = 0.00002
numeric_features = (
                    [length, beam, GT] 
                    # [f'past_{pasthour}hour_{stat}' for stat in stats for pasthour in pasthour_set] + 
                    # [f'past_{pastday}day_same_hour_{stat}' for stat in stats for pastday in pastday_same_hour_set] + 
                    # [f'past_{pastday}day_{stat}' for stat in stats for pastday in pastday_set] + 
                    # [f'currentday_{stat}' for stat in stats]
                    )

categorical_features = [
                        # year,
                        arrived_ship_type,
                        arrived_agent_name, 
                        due_last_port,
                        in_port_imo_no
                        ]

raw_features = numeric_features + categorical_features

