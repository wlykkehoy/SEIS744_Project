This folder contains the code for creating the AI model for
detecting the furnace state from vibration data. 

I tested various types of models; Decision Tree, Support Vector Machine,
and a Neural Network (MLP). The Decision Tree performed the best, thus 
that is the code in this directory.

The data file, ML_20.csv has 2 columns (no header):
  column 1: an indicator of the state the furnace was in
              0 = off
              1 = startup/shutdown
              2 = on
  column 2: vibration reading

When I was collecting data, I found the furnace was a bit more complex
that just being in an 'on' state or 'off' state. On startup, an 'inducer' 
motor comes on for a few minutes, then the furnace comes 'on'. A similar
thing happens on shutdown. The vibration readings from these two 
intermediate states is indistinguishable. Thus I represented them in the
data file with the same value in column 1.

There is both a Jupyter Notebook and plain python code file for each piece
of code. I initially developed the code in a Jypyter Notebook on a Windows PC
using Scikit-learn. There is no way to save and restore Scikit-learn models
across different computer architectures. Thus I saved the Jupyter Notebook
as a python file and re-ran on the Raspberry Pi.
  
     