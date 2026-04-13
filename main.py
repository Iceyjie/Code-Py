import func as fc
import parameter as para

# test

# --------------------------------- Setup ----------------------------------
ship_type = 'CONTAINER' # make prediction and schedules for this ship type;|
train_start_time_line = "2023-01-01"
train_end_time_line = "2025-01-01"
# -------------------------------- End ---------------------------------

# fc.createFolder(para.figure_path)
# fc.mergeDeparted()
data = fc.read_data(ship_type)
ship_data = fc.prepare_ship_data(data, train_start_time_line, train_end_time_line, para.due_last_port)
fc.generate_instance(ship_data)