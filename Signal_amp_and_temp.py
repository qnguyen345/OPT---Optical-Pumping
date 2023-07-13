import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2

# 6.3 - Finding good settings for the bulb temperature
signal_df = pd.read_csv("Data/6_3_data.csv",  index_col=False)
temp = signal_df["Temperature (C)"]
signal_Rb85 = signal_df["Rb85 Optical Signal (V)"]
signal_Rb87 = signal_df["Rb87 Optical Signal (V)"]

def signal_plotter(temp, signal, title):
   title = "Optical Signal vs Temperature ({})".format(title)
   plt.figure(figsize=(10, 8))
   plt.errorbar(temp, signal, yerr=0.01 * np.ones(len(temp)), capsize=5,
                fmt='o', label="Experimental uncertainty $\pm 0.01$ V")
   plt.title(title)
   plt.xlabel("Temperature ($^\circ$C)")
   plt.ylabel("Signal Amplitude (V)")
   plt.grid(alpha=0.3)
   plt.legend()
   plt.show()

# Plots the signal's amplitude vs temp for Rb85 and Rb87
signal_plotter(temp, signal_Rb85, "Rb85")
signal_plotter(temp, signal_Rb87, "Rb87")
