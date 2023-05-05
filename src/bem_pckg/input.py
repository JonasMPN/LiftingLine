# All Inputs from the Assignment 

import numpy as np
import pandas as pd 
import scipy

radius = 50                         # radius of the rotor
n_blades = 3                        # number of blades
inner_radius = 0.2 * radius         # inner end of the blade section
pitch_deg = -2                      # pitch in degrees
pitch = np.radians(pitch_deg)       # pitch angle in radian
yaw_angles = np.radians([0,15,30])  # yaw angles to be calculated in radians

### Operational data 
v_0 = 10                            # [m] Wind speed
tsr = [6,8,10]                      # Tip speed ratios  to be calculated





airfoil = pd.read_excel("../data/polar.xlsx",skiprows=3)    # read in the airfoil. Columns [alpha, cl, cd cm]


### Postprocess 


# with a, r, a' , ft, fn calculate power, thrust 

r_list = np.linspace(inner_radius, radius,10)

def get_element_length():
    """
        Get the element lengths of discrete blade sections
    """
pass

def calculate_thrust(f_n, radial_positions, n_blades, radius):
    """
        Calculate thrust from the normal forces. Account for f_t = 0 at the tip.
    f_n: normal forces 
    radial_positions: radial position along the blade matching the positions of f_n
    n_blades:   number of blades
    radius:     max radius 
    """
    thrust = n_blades  * scipy.integrate.simpson([*f_n,0],[*radial_positions ,radius])

    return thrust

def calculate_power(f_t, radial_positions, n_blades, radius, omega):
    """
        Calculate power from the normal forces. Account for f_n = 0 at the tip.
    f_t: tangential forces 
    radial_positions: radial position along the blade matching the positions of f_n
    n_blades:   number of blades
    radius:     max radius 
    omega:      [rad/s] rotational speed 
    """
    power = omega * n_blades * scipy.integrate.simpson(np.multiply([*radial_positions , radius], [*f_t, 0]),[*radial_positions ,radius])
    return power

def calc_ct(thrust,radius, density, velocity):
    """
        Calculate the thrust coefficient ct 
    """
    return thrust / (0.5*np.pi*(radius**2)*density*(velocity**2))

def calc_cp(power,radius, density, velocity):
    """
        Calculate the power coefficient ct 
    """
    return power / (0.5*np.pi*(radius**2)*density*(velocity**3))

def calc_ct_distribution(f_n, radius, density, velocity): 
    """
    Calculate the distribution of the thrust coefficient along the blade
    f_n: normal forces along the blade
    radius: maximum radius of the Blade
    velocity: fluid velocity od V0
    """
    f_n = np.array(f_n)
    return f_n / (0.5 * np.pi * (radius**2) * density * (velocity**3))

def calc_cp_distribution(f_t, radius, density, velocity):
    """
    Calc the distribution of the power coeff. along the blade
    f_t: tangential forces along the blade
    radius: maximum radius of the Blade
    velocity: fluid velocity od V0
    """
    f_t = np.array(f_t) 
    return f_t / (0.5 * np.pi * (radius**2) * density * (velocity**3))


def _tip_and_root_loss_correction(r: float, tsr: float, rotor_radius: float, root_radius: float, n_blades: int, axial_induction) -> float:
    """
    Returns the factor F for the tip loss correction according to Prandtl
    :param r: current radius
    :param rotor_radius: total radius of the rotor
    :param root_radius: inner radius of the blade 
    :param n_blades: number of blades
    :return: Prandtl tip loss correction
    """
    
    # Tip part
    temp1 = -n_blades/2*(rotor_radius-r)/r*np.sqrt( 1+ ((tsr*r)**2)/((1-axial_induction)**2))
    f_tip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    f_tip[np.isnan(f_tip)] = 0

    # root part
    temp1 = n_blades/2*(rootradius-r)/r*np.sqrt( 1+ ((tsr*r)**2)/((1-axial_induction)**2))
    f_root = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    f_root[np.isnan(f_rppt)] = 0
    return f_root*f_tip, f_tip, f_root


#calculate_thrust([1,2,3],[1,2,3],3,4)
print(calc_ct_distribution([100,200,300,400,400],10,1.25,5))


print("Done!")
