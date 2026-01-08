import os
import numpy as np

from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from send2trash import send2trash

import merope
from statistical_test_func import evaluate_simulation, compare_images, plot_area_distribution

#from Optimization_func import optimize_parameters
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

vtkname = 'double_layer_structure.vtk'

def ThermalAmitex():
    number_of_processors = 6  # per calcoli paralleli
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")



os.chdir('FFT_result')
ThermalAmitex()  

'''
Porosity = 22.315099%
[0.69827388, -0.00094093343, -0.00034271561]
[-0.00094093549, 0.70014452, 0.001222681]
[-0.00034270996, 0.0012226825, 0.70064322]
----------------

'''
