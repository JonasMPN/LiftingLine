import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from bem_pckg.helper_functions import Helper  # problem -> different versions of these helper functions
from bem_pckg import BEM
from bem_pckg import twist_chord
from vortex_system import VortexSystem
from task1 import calc_induction_bem, calc_ll

"""
Do sensitivity analysis on the following parameters:

- Convection speed for the wake
- Blade discretization approach (uniform, cosine)
- Azimuthal discretization ????
- Length of the wake (number  of rotations)
- Vortex core radius (additional)
"""


def task3(wake_speed, blade_discretization, azimuthal_discretization, wake_length):
    # ------------ Get all the inputs -------------#
    radius = 50  # radius of the rotor
    n_blades = 3  # number of blades
    inner_radius = 0.2 * radius  # inner end of the blade section
    pitch_deg = -2  # pitch in degrees
    pitch = np.deg2rad(pitch_deg)  # pitch angle in radian
    resolution_ll = 10  # Spanwise resolution -> seems to break for larger values
    residual_max = 1e-10
    n_iter_max = 1000
    vortex_core_radius = 1
    v_0 = 10
    tsr = 8
    air_density = 1.225
    airfoil = pd.read_excel("../data/polar.xlsx", skiprows=3)  # read in the airfoil. Columns [alpha, cl, cd cm]

    # Loop over different parameters
    fig, ax = plt.subplots(3, 1, figsize=(13, 8))
    for w_s, b_d, a_d, w_l in zip(wake_speed, blade_discretization, azimuthal_discretization, wake_length):
        ll_results = calc_ll(v_0, air_density, tsr, airfoil, radius, n_blades, inner_radius,
                             pitch, resolution_ll, vortex_core_radius, debug=False, wake_length=w_l,
                             disctretization=b_d,
                             residual_max=1e-10, n_iter_max=1000, wake_speed=w_s, resolution_wake=a_d)

        # Plotting
        ax[0].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['a'])
        ax[0].set_ylabel('Axial induction (-)')
        ax[0].grid()

        ax[1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['a_prime'])
        ax[1].set_ylabel('Azimuthal induction (-)')
        ax[1].grid()

        ax[2].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['cl'])
        ax[2].set_xlabel(r'Radial position $\mu$(-)')
        ax[2].set_ylabel(r'$c_l$ (-)')
        ax[2].grid()

    if wake_speed.count(wake_speed[0]) != len(wake_speed):
        plt.legend(wake_speed)
    elif blade_discretization.count(blade_discretization[0]) != len(blade_discretization):
        plt.legend(blade_discretization)
    elif azimuthal_discretization.count(azimuthal_discretization[0]) != len(azimuthal_discretization):
        plt.legend(azimuthal_discretization)
    else:
        plt.legend(wake_length)
    plt.show()

    return


if __name__ == '__main__':

    #########################
    # Sensitivity analysis
    #########################

    # Variable wake speeds
    wake_speed_range = [5, 6, 7, 8, 9, 10]
    blade_discretization = len(wake_speed_range) * ['uniform']
    azimuthal_discretization = len(wake_speed_range) * [50]
    wake_length = len(wake_speed_range) * [1 * 2 * 50]
    task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length)

    # Variable blade discretization
    blade_discretization = ['uniform', 'sin']
    wake_speed_range = len(blade_discretization) * [5]
    azimuthal_discretization = len(blade_discretization) * [50]
    wake_length = len(blade_discretization) * [1 * 2 * 50]
    task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length)

    # Variable azimuthal discretization
    azimuthal_discretization = [25, 50, 75, 100]
    wake_speed_range = len(azimuthal_discretization) * [5]
    blade_discretization = len(azimuthal_discretization) * ['uniform']
    wake_length = len(azimuthal_discretization) * [1 * 2 * 50]
    task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length)

    # Variable wake length
    diameter = 2*50
    wake_length = [1*diameter, 2*diameter, 3*diameter, 4*diameter]
    wake_speed_range = len(wake_length) * [5]
    blade_discretization = len(wake_length) * ['uniform']
    azimuthal_discretization = len(wake_length) * [50]
    task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length)
