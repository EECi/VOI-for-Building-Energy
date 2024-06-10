# Plot EVII curve for GSHP example

from cProfile import label
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


results_file_path = os.path.join('results', 'GSHP_EVII_results.csv')

with open(results_file_path, 'r') as results_file:
    reader = csv.reader(results_file)
    header = next(reader)
    results = np.array(list(reader))

results[0,0] = 0.0 # replace 'EVPI' with 0.0
error_sigmas = results[:,0].astype(float)*100*2 # convert to experimental error as percentage
EVII_values = results[:,1].astype(float)

measurement_costs = [187,1800,5000,10000]

print(error_sigmas, EVII_values)

bc = 'xkcd:cerulean'
fig, ax = plt.subplots()
twinax = ax.twinx()
ax.scatter(error_sigmas, EVII_values/1e3, c='k', zorder=10, clip_on=False, label="EVII")
ax.plot(np.linspace(0,30,100), np.poly1d(np.polyfit(error_sigmas, EVII_values/1e3, 1))(np.linspace(0,30,100)), c='k', zorder=5)
ax.hlines(EVII_values[0]/1e3, 0, 30, colors='grey', linestyles='dashed', zorder=5)
ax.text(15, EVII_values[0]/1e3+0.2, "EVPI", rotation=0, verticalalignment='bottom', horizontalalignment='center', color='grey')
#twinax.plot(error_sigmas[1:], (EVII_values[1:] - measurement_costs)/1e3, c=bc, zorder=10)
twinax.scatter(error_sigmas[1:], (EVII_values[1:] - measurement_costs)/1e3, c=[bc,bc,'r',bc], zorder=10, marker="D", s=22, label="Net value")
ax.annotate("", xy=(error_sigmas[-2], EVII_values[-2]/1e3-0.15), xytext=(error_sigmas[-2], (EVII_values[-2]-measurement_costs[-2])/1e3+0.15),
            arrowprops=dict(
                arrowstyle="<->",
                color="xkcd:grey"
                ),
            zorder=20
    )
ax.text(error_sigmas[-2],(EVII_values[-2]-measurement_costs[-2]/2)/1e3, "Measurement\ncost", rotation=90, verticalalignment='center', horizontalalignment='center', color='xkcd:grey', fontsize='x-small')
plt.xlim(0, 30)
ax.set_xlabel(r"Measurement error of test (%)")
ax.set_ylim(14, 32)
ax.set_ylabel("Expected Value of Imperfect Information, EVII (£k)")
twinax.set_ylim(14, 32)
twinax.set_ylabel("Net value of measurement (£k)")
twinax.tick_params(colors='xkcd:cerulean', axis='y')
twinax.yaxis.label.set_color('xkcd:cerulean')
handlesax, labelsax = ax.get_legend_handles_labels()
handlestwin, labelstwin = twinax.get_legend_handles_labels()
handles = handlesax + handlestwin
labels = labelsax + labelstwin
plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, handletextpad=0.1)
plt.savefig(os.path.join('plots',"GSHP_EVIIs.pdf"), format="pdf", bbox_inches="tight")
plt.show()