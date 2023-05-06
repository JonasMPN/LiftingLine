# Task 1 of lifting line Assignment

from testpck.testfile import testfunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from bem_pckg.helper_functions import Helper
from bem_pckg import BEM
from bem_pckg import twist_chord
from geometry import FrozenWake
import task1
helper  = Helper()


def calc_induction_bem(tsr, pitch, wind_speed = 10, rotor_radius=50, root_radius=10, n_blades=3, density=1.225, resolution=200):
    """
    Function to compute the induction for the whole rotor via BEM
    
    :tsr:           tip speed ratio
    :pitch:         
    :wind_speed:    wind speed far upstream 
    :rotor_radius:  outer radius of the blade
    root_radius:    inner radius of the blade
    :n_blades:      number of blades
    :density:       air density
    :resolution:    resolution used in the bem
    :return:        induction factor a for the whole rotor
    """
    
    bem = BEM.BEM(data_root="../data", file_airfoil="polar.xlsx")  # initialize BEM and set some params
    #bem.set_constants(rotor_radius=rotor_radius, root_radius=50*0.2, n_blades=3, air_density=1.225)
    bem.set_constants(rotor_radius=rotor_radius, root_radius=root_radius, n_blades=n_blades, air_density=density)
    resolution_bem = resolution # spanwise resolution in BEM
    bem.solve_TUD(wind_speed=wind_speed, tip_speed_ratio=tsr, pitch=pitch, resolution=resolution_bem) 
    #bem.solve_TUD(wind_speed=10, tip_speed_ratio=tsr, pitch=-2, resolution=resolution_bem) 
    thrust = bem._calculate_thrust(bem.current_results.f_n, bem.current_results.r_centre)  # compute the thrust from the results via integration
    rotor_area = np.pi * (rotor_radius**2 - root_radius**2)
    C_T = thrust/(1/2 * density * wind_speed**2 * rotor_area) # obtain corresponding thrust coefficient
    # And finally we obtain the induction 
    res = lambda a : 4*a *(1 -a) - C_T # 
    #induction = scipy.optimize.minimize(res,0.2,method ='TNC', bounds=(0,0.4))
    induction = scipy.optimize.newton(res,0.2)
    return induction

def task1():
    """
    Function to combine all operations in task 1 

    That is:
    - computing the induction via bem
    - set up the wake accordingly
    - create the matrix system for the geometrical induction via the vortices
    - solve for circulation
    
    step 2 to 4 need to be implemented still
    """
    #### Get all the inputs 
    radius = 50                         # radius of the rotor
    n_blades = 3                        # number of blades
    inner_radius = 0.2 * radius         # inner end of the blade section
    pitch_deg = -2                      # pitch in degrees
    pitch = np.radians(pitch_deg)       # pitch angle in radian
    resolution = 5                      # -----------> !!!NEEDS TO BE ADAPTED!!!!
    
    ### Operational data 
    v_0 = 10                            # [m] Wind speed
    air_density = 1.225
    tsr = 8                      # Tip speed ratios  to be calculated
    airfoil = pd.read_excel("../data/polar.xlsx",skiprows=3)    # read in the airfoil. Columns [alpha, cl, cd cm]
    
    ### Get twist and chord distributions
    
    ##!!!!!!! Needs to be adapted!
    # Old assignment: from optimized. As this was not mandatory, I assume that something will be given.
    radii = np.linspace(inner_radius, radius, resolution)
    radii_centre_list = [0.5*(radii[i] + radii[i+1]) for i in range(len(radii)-1)] 
    twist_list = [twist_chord.get_twist(r_centre, radius) for r_centre in radii_centre_list]  # Get the twist
    chord_list = [twist_chord.get_chord(r_centre, radius) for r_centre in radii_centre_list]  # Get the twist
    # So now we have defined the rotor fully. 
    # We now want to run BEM to obtain a CT value which we can use to compute the wake convection
   

    
    # With the induction the 
    induction = calc_induction_bem(tsr,-2)

    ######################################################
    ################ PART 2
    ######################################################
    
    # Compute the wake structure
    
    # compute inputs for the wake tool:
    omega = tsr*v_0/radius
    wake_speed = (1-induction)*v_0  # take the speed at the rotor or the speed far downstream? Probably at the rotor is a closer guess

    wake = FrozenWake()
    wake.set_rotor(0.2+np.linspace(0,1,5), np.linspace(0,0.2,5)[::-1], blade_rotation=-0.5*np.pi/2,
                   rotor_rotation_speed=omega)
    #wake.set_wake_properties(wake_speed=0.5, wake_length=5, time_resolution=50)
    wake.set_wake_properties(wake_speed=wake_speed, wake_length=5, time_resolution=50)
    wake.rotor()
    wake.blade_elementwise_visualisation()


if __name__=="__main__":
    task1()
