This folder contains the Python code to collect and dump
vibration data. It reflects the algorithm and parameters
that, after lots of experimentation, seems to give good 
results (data that an ML model can leverage well).

Note the code depends on the Adafruit python library for
the ADXL345 acceleromter. That library can be installed
as follows:

    sudo pip3 install adafruit-circuitpython-ADXL345

Also, make sure the Raspberry Pi's I2C interface is enabled
and that the Pi can see the ADXL345; it should show up
on channel 53:

    sudo i2cdetect -y 1

Here is a great tutorial:

    https://pimylifeup.com/raspberry-pi-accelerometer-adxl345/

Link to sensor information on the Adafruit website:

    https://learn.adafruit.com/adxl345-digital-accelerometer

Link to the Adafruit python library documentation:

    https://circuitpython.readthedocs.io/projects/adxl34x/en/latest/index.html

Link to the Adafruit python library github repo:

    https://github.com/adafruit/Adafruit_CircuitPython_ADXL34x/blob/master/adafruit_adxl34x.py