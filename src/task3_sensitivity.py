import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from bem_pckg.helper_functions import Helper  # problem -> different versions of these helper functions
from bem_pckg import BEM
from bem_pckg import twist_chord
from vortex_system import VortexSystem
from task1 import calc_induction_bem, calc_lift, calc_circulation, calc_velocity


"""
Do sensitivity analysis on the following parameters:

- Convection speed for the wake
- Blade discretization approach (uniform, cosine)
- Azimuthal discretization ????
- Length of the wake (number  of rotations)
- Vortex core radius (additional)
"""


def task3(wake_speed : np.array, blade_discretization, azimuthal_discretization, wake_length):


    # ------------ Get all the inputs -------------#
    radius = 50  # radius of the rotor
    n_blades = 3  # number of blades
    inner_radius = 0.2 * radius  # inner end of the blade section
    pitch_deg = -2  # pitch in degrees
    pitch = np.deg2rad(pitch_deg)  # pitch angle in radian
    resolution = 14  # Spanwise resolution -> seems to break for larger values
    residual_max = 1e-10
    n_iter_max = 1000
    vortex_core_radius = 1
    u_inf = 10
    tsr = 8
    air_density = 1.225

    # Get induction from BEM
    induction, results_bem = calc_induction_bem(tsr,pitch_deg)
    u_rotor = u_inf * (1-induction)
    print('BEM done')


    # Loop over different parameters
    for w_s, b_d, a_d, w_l in zip(wake_speed, blade_discretization, azimuthal_discretization, wake_length):

        # Check the blade_discretization
        if b_d == "uniform":
            radii_ends = np.linspace(inner_radius, radius, resolution) # uniform distribution
        elif b_d == "cosine":
            radii_ends = (np.sin(np.linspace(-np.pi / 2, np.pi / 2, resolution)) / 2 + 0.5) * (
                        radius - inner_radius) + inner_radius  # sine distribution
        else:
            print('Only uniform  and cosine distributions are accepted')

        chord_ends = twist_chord.get_chord(radii_ends, radius) # chord at the ends of each section
        twist_ends = -twist_chord.get_twist(radii_ends, radius)
        pitch = -pitch






    # if blade_discretization:
    # if azimuthal_discretization:
    # if wake_length:
    # if core_radius:
    return

if __name__ == '__main__':
