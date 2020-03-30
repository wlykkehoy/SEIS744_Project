# Import libraries that we will be using
import sys
import time
import collections
import statistics
import numpy as np
import pandas as pd
import pickle

import board
import busio
import adafruit_adxl34x

from azure.iot.device import IoTHubDeviceClient, Message


# Some constant values; change these to change behavior of the app
READINGS_PER_SECOND = 20
RUNNING_AVG_WINDOW_SIZE = 20
IOT_HUB_CONNECTION_STRING = "HostName=SEIS744Project.azure-devices.net;DeviceId=RazPi;SharedAccessKey=4lJeOPKGVS086AlDY/l2l0RYH7bABXhRvpk1THe7j+w="
MSG_TEMPLATE = '{{"ts":{ts_data},"vib":{vib_data:10.8f},"state":{state_data}}}'



def main_loop(verbose):
    # Create an IoT Hub client
    client = IoTHubDeviceClient.create_from_connection_string(IOT_HUB_CONNECTION_STRING)

    # Initialize the accelerometer
    i2c = busio.I2C(board.SCL, board.SDA)
    accel = adafruit_adxl34x.ADXL345(i2c)
    
    # Load the AI model to determine (predict) furnace state
    state_pred_model = pickle.load(open('tree_pipeline.pkl', 'rb')) 
    
    # Calc the amount of time (sec) to sleep between readings
    sleep_time = 1.0 / READINGS_PER_SECOND
    
    # We will be doing a running average of the readings over a given window size
    running_avg_window = collections.deque(maxlen=RUNNING_AVG_WINDOW_SIZE)
        
    # Initilize prevoius reading variables, else will start off with undefined or zero deltas
    prev_x, prev_y, prev_z = accel.acceleration
    time.sleep(sleep_time)
    
    # Enter an infinite loop taking readings, determining state, and pushing data
    #  up to the Azure IoT Hub
    try: 
        while True:
            # Take an accelerometer reading
            reading_time = time.time()
            new_x, new_y, new_z = accel.acceleration
            
            # Turn the accelerometer reading into an indicator of vibration
            delta_x = new_x - prev_x
            delta_y = new_y - prev_y
            delta_z = new_z - prev_z
            
            sum_delta = abs(delta_x) + abs(delta_y) + abs(delta_z)
            
            # Find the running average
            running_avg_window.append(sum_delta)
            running_avg = statistics.mean(running_avg_window)
        
            # Use our AI model to determine (predict) the state
            X = np.array([[running_avg]])
            state = state_pred_model.predict(X)
            #if (verbose):
            #    print('{}, {:>10.6f}'.format(state[0], running_avg), flush=True)
        
            # Format the data into JSON
            msg_text = MSG_TEMPLATE.format(ts_data=reading_time, vib_data=running_avg, state_data=state[0])
            if (verbose):
                print(msg_text, flush=True)
            
            # Package the message up and send on to the Azure IoT Hub
            msg = Message(msg_text)
            client.send_message(msg)          
            
            # Get ready for next reading
            prev_x = new_x
            prev_y = new_y
            prev_z = new_z
        
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        # Just fall back to the main
        print('... Ctrl-C detected ...', flush=True)


if (__name__ == '__main__'):
    print ( "FurnaceMonitor started...", flush=True )
    print ( "Press Ctrl-C to stop", flush=True )
    main_loop(verbose=True)
    print ( "FurnaceMonitor stopped", flush=True )
