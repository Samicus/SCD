import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
#from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
#from optuna.visualization.matplotlib import plot_slice
from slice_plot import plot_slice
from matplotlib import pyplot as plt
from parallel_coordinate import plot_parallel_coordinate
from heatmap import plot_heatmap


study_name = 'params'
storage = 'sqlite:////home/samnehme/Dev/SCD_project/SCD/param_plots/params.db'
study = optuna.load_study(study_name, storage)

"""
plt.figure()
plot_contour(study)
plt.savefig("Contour.png")
plt.figure()
plot_edf(study)
plt.savefig("EDF.png")
plt.figure()
plot_intermediate_values(study)
plt.savefig("Intermediate.png")
plt.figure()
plot_optimization_history(study)
plt.savefig("Optimization_History.png")
plt.figure()
plot_parallel_coordinate(study)
plt.savefig("Parallell_Coordinate.png")
plt.figure()
plot_param_importances(study)
plt.savefig("Param_Importance.png")
plt.figure()
plot_slice(study)
plt.savefig("Slice.png")
"""
#plot_slice(study)
#plt.figure()
plot_heatmap(study)
plt.show()