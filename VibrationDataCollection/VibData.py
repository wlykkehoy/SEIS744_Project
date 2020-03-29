import time
import collections
import statistics

import board
import busio
import adafruit_adxl34x

# Change this to flag the state the furnace is in
FURNACE_STATE = 0    # 0 => off; 1  => starting; 2 => on; 3 => stopping

# Some constant values; change for experimenting
READINGS_PER_SECOND = 20
RUNNING_AVG_WINDOW_SIZE = 20
NUM_CASES_TO_COLLECT = 1000



sleep_time = 1.0 / READINGS_PER_SECOND

# Initialize the accelerometer
i2c = busio.I2C(board.SCL, board.SDA)
accel = adafruit_adxl34x.ADXL345(i2c)

# Do a running average of the readings
running_avg_window = collections.deque(maxlen=RUNNING_AVG_WINDOW_SIZE)

# Initilize prevoius reading variables, else will start off with undefined deltas
prev_x, prev_y, prev_z = accel.acceleration
time.sleep(sleep_time)

# Loop and dump readings to stdout
case_count = 0
while case_count < NUM_CASES_TO_COLLECT:
    new_x, new_y, new_z = accel.acceleration

    delta_x = new_x - prev_x
    delta_y = new_y - prev_y
    delta_z = new_z - prev_z

    sum_delta = abs(delta_x) + abs(delta_y) + abs(delta_z)
    
    running_avg_window.append(sum_delta)
    running_avg = statistics.mean(running_avg_window)

    print('{}, {:>10.6f}'.format(FURNACE_STATE, running_avg))
    case_count += 1
   
    prev_x = new_x
    prev_y = new_y
    prev_z = new_z
    
    time.sleep(sleep_time)


