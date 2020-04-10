# Import libraries that we will be using
import sys
import getopt
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
RUNNING_VIB_WINDOW_SIZE = 20      # num readings to average over for our vibration calc; 20 => a 1 second window
RUNNING_STATE_WINDOW_SIZE = 100   # num readings to consider for determing current state; 100 => a 5 second window

READINGS_PER_IOT_HUB_MSG = 20     # Thus will be pumping 1 message/sec to IoT Hub
IOT_HUB_CONNECTION_STRING = "HostName=seis744-project-hub.azure-devices.net;DeviceId=razpi;SharedAccessKey=jXQ3aV44XyohPTYced9rrJSgQe56d0wumgl5m/9Xhxw="
#MSG_TEMPLATE = '{{"ts":"{ts_data}","vib":{vib_data:10.8f},"state":"{state_data}"}}'
MSG_TEMPLATE = '{{"ts":"{ts_data}","vib":{vib_data:10.8f},"state":"{state_data}","sis":"{seconds_in_state}"}}'



def main_loop(echo_messages, echo_sum_delta, echo_running_window):
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
    running_vib_window = collections.deque(maxlen=RUNNING_VIB_WINDOW_SIZE)
    
    # And also doing a window over the predicted state, taking the most predicted value
    running_state_window = collections.deque(maxlen=RUNNING_STATE_WINDOW_SIZE)

    # We will be tracking state changes and time the state changed
    current_state = -99    # an invalid value
    state_change_time = 0
    
    # Initilize prevoius reading variables, else will start off with undefined or zero deltas
    prev_x, prev_y, prev_z = accel.acceleration
    time.sleep(sleep_time)
    
    # Enter an infinite loop taking readings, determining state, and pushing data
    #  up to the Azure IoT Hub
    reading_count = 0
    try: 
        while True:
            # Take an accelerometer reading
            #reading_time = time.gmtime()
            reading_time = time.time()
            new_x, new_y, new_z = accel.acceleration
            reading_count += 1
            
            # Turn the accelerometer reading into an indicator of vibration
            delta_x = new_x - prev_x
            delta_y = new_y - prev_y
            delta_z = new_z - prev_z
            
            sum_delta = abs(delta_x) + abs(delta_y) + abs(delta_z)
            if (echo_sum_delta):
                print('{:>10.6f}'.format(sum_delta), flush=True)
            
            # Find the running average
            running_vib_window.append(sum_delta)
            vib_avg = statistics.mean(running_vib_window)
        
            # Use our AI model to determine (predict) the state, using the most predicted
            #  state over a running window as the 'true' current state
            X = np.array([[vib_avg]])
            pred_state = state_pred_model.predict(X)
            running_state_window.append(pred_state)
            state_vals, state_counts = np.unique(running_state_window, return_counts=True)
            state = state_vals[np.argmax(state_counts)]
                        
            if (echo_running_window):
                print('{}, {:>10.6f}'.format(state, vib_avg), flush=True)
        
            # If there is a state change, update vars
            if (state != current_state):
                current_state = state
                state_change_time = reading_time
        
            if (reading_count == READINGS_PER_IOT_HUB_MSG):
                # Format the data into JSON
                formatted_time = time.strftime('%Y-%m-%dT%H:%M:%Sz', time.gmtime(reading_time))
                seconds_in_state = int(reading_time - state_change_time)
                msg_text = MSG_TEMPLATE.format(ts_data=formatted_time, vib_data=vib_avg,
                                               state_data=state, seconds_in_state=seconds_in_state)
                if (echo_messages):
                    print(msg_text, flush=True)
            
                # Package the message up and send on to the Azure IoT Hub
                msg = Message(msg_text)
                client.send_message(msg)
                
                reading_count = 0
                
            
            # Get ready for next reading
            prev_x = new_x
            prev_y = new_y
            prev_z = new_z
        
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        # Just fall back to the main
        print('... Ctrl-C detected ...', flush=True)


if (__name__ == '__main__'):
    # Initialize some flags to control output
    echo_messages = False
    echo_sum_delta = False
    echo_running_window = False
    
    # Pull off command line args & set flags
    try:
        args, _ = getopt.getopt(sys.argv[1:], 'msr')
    except getopt.GetoptError:
        print('Usage: {} [-m] [-s] [-r]'.format(sys.argv[0]))
        sys.exit(2)

    for opt, _ in args:      
        if (opt == '-m'):
            echo_messages = True
        elif (opt == '-s'):
            echo_sum_delta = True
        elif (opt == '-r'):
            echo_running_window = True

    print ( "FurnaceMonitor started...", flush=True )
    print ( "Press Ctrl-C to stop", flush=True )
    main_loop(echo_messages, echo_sum_delta, echo_running_window)
    print ( "FurnaceMonitor stopped", flush=True )
