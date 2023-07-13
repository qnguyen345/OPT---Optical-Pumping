import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2
plt.rcParams.update({'font.size': 16})

# Reads file
res_df = pd.read_csv("Data/6_4_data.csv",  index_col=False)

def res_plotter(res_df, neg=True, pos=True, mag_field=False):
   # Data
   curr = res_df["current (A)"]
   freq_Rb85 = res_df["resonance freq Rb85 (MHz)"]
   freq_Rb87 = res_df["resonance freq Rb87 (MHz)"]
   b_field_Rb85 = res_df["B field Rb85 (Gaussian)"]
   b_field_Rb87 = res_df["B field Rb87 (Gaussian)"]
   yerr = res_df["error (kHz)"] / 1000

   # Filtering data
   plt.figure(figsize=(10, 8))
   if mag_field:
      title = "Resonance Frequency vs Magnetic Field"
      x_label = "Magnetic Field (G)"
      x_var = "B"
      x_data_Rb85 = b_field_Rb85
      x_data_Rb87 = b_field_Rb87
   else:
      title = "Resonance Frequency vs {}Current"
      x_label = "Current (A)"
      x_var = "i"
      x_data_Rb85 = curr
      x_data_Rb87 = curr

   if neg and not pos:
      title = title.format("Negative ")
      fit_label_Rb85 = "Rb85 fit: $f$ = {:.3f}*${} -${:.3f}"
      fit_label_Rb87 = "Rb87 fit: $f$ = {:.3f}*${} -${:.3f}"
      x_data_Rb85 = [i for i in x_data_Rb85 if i < 0]
      x_data_Rb87 = x_data_Rb85
      freq_Rb85 = freq_Rb85[:len(x_data_Rb85)]
      freq_Rb87 = freq_Rb87[:len(x_data_Rb85)]
      yerr = yerr[:len(x_data_Rb85)]

      # Linear fit
      def lin_fit(x, a, b):
         x = np.array(x)
         return a * x + b

   elif pos and not neg:
      title = title.format("Positive ")
      fit_label_Rb85 = "Rb85 fit: $f$ = {:.3f}*${} +${:.3f}"
      fit_label_Rb87 = "Rb87 fit: $f$ = {:.3f}*${} +${:.3f}"
      x_data_Rb85 = [i for i in x_data_Rb85 if i >= 0]
      x_data_Rb87 = x_data_Rb85
      freq_Rb85 = freq_Rb85[-len(x_data_Rb85):]
      freq_Rb87 = freq_Rb87[-len(x_data_Rb85):]
      yerr = yerr[-len(x_data_Rb85):]

      #Linear fit
      def lin_fit(x, a, b):
         x = np.array(x)
         return a * x + b

   else:
      title = title.format("")
      fit_label_Rb85 = "Rb85 fit: $f$ = |{:.3f}*${} +${:.3e}|"
      fit_label_Rb87 = "Rb87 fit: $f$ = |{:.3f}*${} +${:.3e}|"

      def lin_fit(x, a, b):
         x = np.array(x)
         return abs(a * x + b)

   # Curve fitting
   x_fit_Rb85 = np.linspace(min(x_data_Rb85), max(x_data_Rb85), 100)
   x_fit_Rb87 = np.linspace(min(x_data_Rb87), max(x_data_Rb87), 100)
   opt_Rb85, pcov_Rb85 = curve_fit(lin_fit, x_data_Rb85, freq_Rb85)
   opt_Rb87, pcov_Rb87 = curve_fit(lin_fit, x_data_Rb87, freq_Rb87)
   fit_err_Rb85 = np.sqrt(np.diag(pcov_Rb85))
   fit_err_Rb87 = np.sqrt(np.diag(pcov_Rb87))

   # C^2, Goodness of fit
   def chisquared(obs, model, error):
      return sum(((obs - model) / error) ** 2)

   N = len(freq_Rb85)
   chisq85 = chisquared(freq_Rb85, lin_fit(x_data_Rb85, *opt_Rb85),
                        opt_Rb85[1] * N)
   p85 = 1 - chi2.cdf(chisq85, N - 2)
   chisq87 = chisquared(freq_Rb87, lin_fit(x_data_Rb87, *opt_Rb87),
                        opt_Rb87[1] * N)
   p87 = 1 - chi2.cdf(chisq87, N - 2)

   # Plotting
   plt.errorbar(x_data_Rb85, freq_Rb85, yerr=yerr, label="Rb85", fmt='ro',
                capsize=5)
   plt.errorbar(x_data_Rb87, freq_Rb87, yerr=yerr, label="Rb87", fmt='bo',
                capsize=5)
   plt.plot(x_fit_Rb85, lin_fit(x_fit_Rb85, *opt_Rb85), 'r-',
            label=fit_label_Rb85.format(opt_Rb85[0], x_var, abs(opt_Rb85[1])))
   plt.plot(x_fit_Rb87, lin_fit(x_fit_Rb87, *opt_Rb87), 'b-',
            label=fit_label_Rb87.format(opt_Rb87[0], x_var, abs(opt_Rb87[1])))
   plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
   plt.title(title)
   plt.xlabel(x_label)
   plt.ylabel("Frequency (MHz)")
   plt.grid(alpha=0.3)
   plt.legend()
   plt.show()

   # Return values
   results = {"Rb85 fit": opt_Rb85,
              "Rb85 fit error": fit_err_Rb85,
              "Rb85 p value": p85,
              "Rb85 chisquared": chisq85,
              "Rb87 fit": opt_Rb87,
              "Rb87 fit error": fit_err_Rb87,
              "Rb87 p value": p87,
              "Rb87 chisquared": chisq87}
   return results

curr_fit = res_plotter(res_df, neg=True, pos=True)
curr_fit
neg_fit = res_plotter(res_df, neg=True, pos=False)
neg_fit
pos_fit = res_plotter(res_df, neg=False, pos=True)
pos_fit
neg_mag_fit = res_plotter(res_df, neg=True, pos=False, mag_field=True)
neg_mag_fit
pos_mag_fit = res_plotter(res_df, neg=False, pos=True, mag_field=True)
pos_mag_fit