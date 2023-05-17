# Task 1 of lifting line Assignment

from testpck.testfile import testfunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from bem_pckg.helper_functions import Helper # problem -> different versions of these helper functions 
from bem_pckg import BEM
from bem_pckg import twist_chord
from vortex_system import VortexSystem
import task1
helper  = Helper()


def calc_induction_bem(tsr, pitch, wind_speed = 10, rotor_radius=50, root_radius=10, n_blades=3, density=1.225, resolution=200):
    """
    Function to compute the induction for the whole rotor via BEM from the last assignment
    
    
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
    # Now we need the thrust coefficient to get the 
    rotor_area = np.pi * (rotor_radius**2 - root_radius**2)
    C_T = thrust/(1/2 * density * wind_speed**2 * rotor_area) # obtain corresponding thrust coefficient
    # And finally we obtain the induction from solving the CT - induction equation for the induction
    res = lambda a : 4*a *(1 -a) - C_T # 
    #induction = scipy.optimize.minimize(res,0.2,method ='TNC', bounds=(0,0.4))
    induction = scipy.optimize.newton(res,0.2)
    return induction

def calc_lift(aoa: np.array, chord: np.array, inflow_speed: np.array, rho : float = 1.225,
              path_to_polar: str="../data/polar.xlsx") -> np.array:
    """ 
    Function to compute the lift force per unit span along the blade.

    Note that the inflow speed needs to include the rotation!
    
    :aoa:           Angles of attack in radian
    :inflow_speed:  Corresponding inflow speeds
    :path_to_polar: Path of the polar data used to obtain the force coefficients
    """
    polar_data = pd.read_excel(path_to_polar, skiprows=3) # read in polar data

    cl_function = scipy.interpolate.interp1d(polar_data["alpha"], polar_data["cl"]) 
    cl = cl_function(np.rad2deg(aoa))# get cl along the blade
    lift = 0.5 * rho * inflow_speed**2 * cl  
    return lift

def calc_circulation(lift: np.array, u_inflow: float, old_circulation: np.array, rho:float=1.225 ) -> np.array:
    """
    Compute the circulation at every control point based on Kutta-Joukowsky

    :lift:              Lift per unit span for every section of the blade
    :old_circulation:   Old circulation 
    :u_infty:           Local velocity magnitude 
    :rho:               Fluid density 
    """

    circulation = (old_circulation  +  lift / (rho * u_inflow)) / 2
    return circulation

def calc_velocity(u_infty: float, omega: float, 
                  radial_positions: np.array, u_induced :np.array, v_induced: np.array ) -> dict:
    """
    Compute the velocity components as the sum of far field velocity, rotation, and the induced velocity field and return a dict.
    """
    #breakpoint()
    u = u_infty + u_induced     # x-component
    v = omega * radial_positions + v_induced   # y - component
    velocity_magnitude = np.sqrt(u**2 + v**2) # absolute
    return {"u": u, "v": v, "magnitude":velocity_magnitude} # return a dict!

def task1(debug=False):
    """
    Function to combine all operations in Task 1 

    That is:
    - computing the induction via bem
    - set up the wake accordingly
    - create the matrix system for the geometrical induction via the vortices
    - solve for circulation iteratively
    
    step 4 needs debugging
    """
    
    #### Get all the inputs 
    radius = 50                         # radius of the rotor
    n_blades = 3                        # number of blades
    inner_radius = 0.2 * radius         # inner end of the blade section
    pitch_deg = -2                      # pitch in degrees
    pitch = np.radians(pitch_deg)       # pitch angle in radian
    resolution = 5                      # -----------> !!!NEEDS TO BE ADAPTED!!!!
    residual_max = 10**-5
    n_iter_max = 100
    
    ### Operational data 
    v_0 = 10                            # [m] Wind speed
    air_density = 1.225
    tsr = 8                      # Tip speed ratios  to be calculated
    airfoil = pd.read_excel("../data/polar.xlsx",skiprows=3)    # read in the airfoil. Columns [alpha, cl, cd cm]
    
    ### Get twist and chord distributions
    
    ##!!!!!!! Needs to be adapted!
    # Get the positions along the span for each element and the nodes
    # Compute discretization of the blade:
    # uniform distribution (for now)
    radii = np.linspace(inner_radius, radius, resolution) # radial positions of the ends of each section
    radii_centre_list = np.array([0.5*(radii[i] + radii[i+1]) for i in range(len(radii)-1)] )  # centre of sections
    twist_list_centre = np.array([twist_chord.get_twist(r_centre, radius) for r_centre in radii_centre_list])  # Get the twist in radian at the centers
    chord_list_centre = np.array([twist_chord.get_chord(r_centre, radius) for r_centre in radii_centre_list] ) # Get the chord at the centers
     
    # changed to taking at the edges of an element.... if that really makes sense needs to be discussed
    twist_list = np.array( [twist_chord.get_twist(r, radius) for r in radii])  # Get the twist at boundary locations
    chord_list = np.array([twist_chord.get_chord(r, radius) for r in radii])  # Get the chord
    # So now we have defined the rotor fully. 
    # We now want to run BEM to obtain a CT value which we can use to compute the wake convection
   
    #breakpoint()
    
    # With the induction the 
    induction = calc_induction_bem(tsr,-2)
    print("BEM done")
    #------------------------------------------------------#
    # PART 2 - set up wake system
    #------------------------------------------------------#
    
    # Compute the wake structure
    
    # compute inputs for the wake tool:
    omega = tsr*v_0/radius
    wake_speed = (1-induction)*v_0  # take the speed at the rotor or the speed far downstream? Probably at the rotor is a closer guess
    wake_length = 1 * 2 * radius # how many diameters should the wake have?
    time_resolution = 300

    
    # initializy wake geometry
    vortex_system = VortexSystem()
    vortex_system.set_blade(radii, chord_list, blade_rotation=twist_list + pitch, 
                            rotor_rotation_speed=omega) # The twist list at the boundaries should be required only for the computation of the wake -> which way the 
    #wake.set_wake_properties(wake_speed=0.5, wake_length=5, time_resolution=50)
    # set the properties of the wake. note that the resolution here should be something related to the discretization of the trailing vortices rather than the discretization of the blade
    vortex_system.set_wake(wake_speed=wake_speed, wake_length=wake_length, resolution=time_resolution)

    vortex_system.rotor() # create the vortex system
    if debug:
        vortex_system.blade_elementwise_visualisation()

    #------------------------------------------------------#
    # PART 3 - Compute matrices
    #------------------------------------------------------#
    # 3.1 Set Control points
    print("Create vortex system")
    vortex_system.set_control_points(x_control_points=np.multiply(1/4*chord_list_centre,  np.cos(twist_list_centre + pitch)),
                                     y_control_points=np.multiply(1/4*chord_list_centre,  np.sin(twist_list_centre + pitch)),
                                     z_control_points=radii_centre_list)

    # 3.2 Create the matrices connecting the circulation and the velocity field

    print("Create induction matrices")
    trailing_mat_u, trailing_mat_v, trailing_mat_w = vortex_system.trailing_induction_matrices() # calculate the trailing induction matrices
    bound_mat_u, bound_mat_v, bound_mat_w = vortex_system.bound_induction_matrices() # calculate the bound induction matrices
    
    #------------------------------------------------------#
    # PART 4 - Iterate to find the right circulation
    #------------------------------------------------------#
    
    print("Initialize values")
    # 4.1 Initialize the arrays for the velocity
    u_induced = np.zeros(len(radii_centre_list))
    v_induced = np.zeros(len(radii_centre_list))
    inflow_velocity = calc_velocity(v_0, omega, radii_centre_list, u_induced, v_induced) # we now have the u,v velocity vector
    # We can compute the effective angle of attack with it
    # -> This needs to be changed -> shouldnt use induced velocity for that
   # effective_aoa = twist_list_centre + np.rad2deg(np.arctan(-v_induced/u_induced)) # second part should be 0 here, as were not yet inducing velocities
    #effective_aoa = twist_list_centre + np.rad2deg(np.arctan(-inflow_velocity["v"]/inflow_velocity["u"])) # second part should be 0 here, as were not yet inducing velocities
    #effective_aoa = twist_list_centre + (np.arctan(-inflow_velocity["v"]/inflow_velocity["u"])) # second part should be 0 here, as were not yet inducing velocities -----> in radian
    effective_aoa = - twist_list_centre + (np.arctan(inflow_velocity["u"]/inflow_velocity["v"])) # second part should be 0 here, as were not yet inducing velocities -----> in radian
    #breakpoint()
    lift = calc_lift(effective_aoa, chord_list_centre, inflow_velocity["magnitude"])
   
    # add relaxation
    # from the lift we can obtain the bound circulation
    bound_circulation = calc_circulation(lift, inflow_velocity["magnitude"], np.array([0]*len(lift))) # compute with the magnitude -> is that so exact ? 
    # The trailing circulation is the delta between two bound circulations, or the "step"
    #breakpoint()
    trailing_circulation = np.append(0,bound_circulation) - np.append (bound_circulation, 0)  
    #trailing_circulation = np.reshape(trailing_circulation, (radii.size,1))

    # Everything is defined, now create a loop to iterate over the circulations
    residual = 1
    n_iter = 0
    L_mag_new = 0
    while residual > residual_max and n_iter<=n_iter_max:
        L_mag = L_mag_new # magnitude of the lift
        u_induced = trailing_mat_u @ trailing_circulation + bound_mat_u @ bound_circulation # calculate stream-wise induction
        v_induced = trailing_mat_v @ trailing_circulation + bound_mat_v @ bound_circulation # calc azimuthal velocity / downwash
        inflow_velocity = calc_velocity(v_0, omega, radii_centre_list, u_induced, v_induced) # we now have the u,v velocity vector
        U = inflow_velocity["u"] 
        V = inflow_velocity["v"] 
        #breakpoint()
        #effective_aoa = twist_list_centre + np.arctan(-inflow_velocity["v"]/inflow_velocity["u"]) 
        effective_aoa = -twist_list_centre + pitch + np.arctan(inflow_velocity["u"]/inflow_velocity["v"]) 
        lift = calc_lift(np.deg2rad(effective_aoa), chord_list_centre, inflow_velocity["magnitude"])
        # from the lift we can obtain the bound circulation
        bound_circulation = calc_circulation(lift, inflow_velocity["magnitude"],bound_circulation) # compute with the magnitude -> is that so exact ? 
        #bound_circulation = (bound_circulation )/2

        trailing_circulation = np.append(0,bound_circulation) - np.append (bound_circulation, 0)  
        #trailing_circulation = np.reshape(trailing_circulation, (radii.size,1))
        
        L_mag_new = np.sum(lift)
        residual = np.abs(L_mag - L_mag_new) 
        # update circulation
        print(f"Iter: {n_iter} \t residual: {residual}")
        n_iter +=1
    fig, axs = plt.subplots(5,1)
    axs[0].plot(radii_centre_list, bound_circulation)
    axs[1].plot(trailing_circulation)
    axs[2].plot(radii_centre_list, lift)
    axs[3].plot(radii_centre_list, u_induced)
    axs[4].plot(radii_centre_list, v_induced)
    plt.show()

if __name__=="__main__":
    task1(debug=False)
