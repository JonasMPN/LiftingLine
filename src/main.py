# Description


# Imports
from testpck.testfile import testfunc
import numpy as np
import matplotlib.pyplot as plt
from bem_pckg.helper_functions import Helper
from bem_pckg import BEM
from bem_pckg import twist_chord

helper  = Helper()


task_1 = True


# Task 1 : Run for the same conditions as the BEM
if task_1:
    #### Get all the inputs 
    radius = 50                         # radius of the rotor
    n_blades = 3                        # number of blades
    inner_radius = 0.2 * radius         # inner end of the blade section
    pitch_deg = -2                      # pitch in degrees
    pitch = np.radians(pitch_deg)       # pitch angle in radian
    resolution = 5                      # -----------> !!!NEEDS TO BE ADAPTED!!!!
    ### Operational data 
    v_0 = 10                            # [m] Wind speed
    tsr = [6,8,10]                      # Tip speed ratios  to be calculated
    airfoil = pd.read_excel("../data/polar.xlsx",skiprows=3)    # read in the airfoil. Columns [alpha, cl, cd cm]
    
    ### Get twist and chord distributions
    
    ##!!!!!!! Needs to be adapted!
    # Old exam: from optimized. As this was not mandatory, I assume that something will be given.
    radii = np.linspace(inner_radius, radius, resolution)
    radii_centre_list = [0.5*(radii[i] + radii[i+1]) for i in range(len(radii)-1)] 
    twist_list = [twist_chord.get_twist(r_centre, radius) for r_centre in radii_centre_list]  # Get the twist
    chord_list = [twist_chord.get_chord(r_centre, radius) for r_centre in radii_centre_list]  # Get the twist
    
    # So now we have defined the rotor fully. 
    # We now want to run BEM to obtain a CT value which we can use to compute the wake convection


