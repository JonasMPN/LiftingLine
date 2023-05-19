# Task 1 of lifting line Assignment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from bem_pckg.helper_functions import Helper # problem -> different versions of these helper functions 
from bem_pckg import BEM
from bem_pckg import twist_chord
from vortex_system import VortexSystem
import task1
helper = Helper()


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
    # res = lambda a : 4*a *(1 -a) - C_T #
    # induction = scipy.optimize.minimize(res,0.2,method ='TNC', bounds=(0,0.4))
    # induction = scipy.optimize.newton(res,0.2)
    return 1/2*(1-np.sqrt(1-C_T)), bem.current_results  # analytical induction factor solution

def calc_lift(aoa: np.ndarray, chord: np.ndarray, inflow_speed: np.ndarray, rho : float = 1.225,
              path_to_polar: str="../data/polar.xlsx") -> np.ndarray:
    """ 
    Function to compute the lift force per unit span along the blade.

    Note that the inflow speed needs to include the rotation!
    
    :aoa:           Angles of attack in radian
    :inflow_speed:  Corresponding inflow speeds
    :path_to_polar: Path of the polar data used to obtain the force coefficients
    """
    polar_data = pd.read_excel(path_to_polar, skiprows=3) # read in polar data

    cl_function = scipy.interpolate.interp1d(polar_data["alpha"], polar_data["cl"]) 
    cd_function = scipy.interpolate.interp1d(polar_data["alpha"], polar_data["cd"]) 
    cl = cl_function(np.rad2deg(aoa))# get cl along the blade
    cd = cd_function(np.rad2deg(aoa))# get cl along the blade
    lift = 0.5*rho*chord*inflow_speed**2*cl
    return lift, cl, cd

def calc_circulation(lift: np.ndarray,
                     u_inflow: float or np.ndarray,
                     old_circulation: np.ndarray,
                     rho:float=1.225 ) -> np.ndarray:
    """
    Compute the circulation at every control point based on Kutta-Joukowsky. Uses some under-relaxation.

    :lift:              Lift per unit span for every section of the blade
    :old_circulation:   Old circulation 
    :u_infty:           Local velocity magnitude 
    :rho:               Fluid density 
    """
    return (old_circulation+lift/(rho*u_inflow))/2


def calc_circulation_from_cl(cl: np.array,
                             c: np.array,
                             a: np.array,
                             u_inf: float):
    """
    
    """
    breakpoint()
    gamma = cl * 0.5 * c * (1-a) * u_inf
    return gamma


def calc_velocity(u_inf: float, omega: float,
                  radial_positions: np.ndarray, u_induced: np.ndarray, v_induced: np.ndarray ) -> dict:
    """
    Compute the velocity components as the sum of far field velocity, rotation, and the induced velocity field and return a dict.
    """
    u = u_inf+u_induced     # x-component
    v = omega*radial_positions+v_induced   # y - component
    velocity_magnitude = np.sqrt(u**2 + v**2) # absolute
    return {"u": u, "v": v, "magnitude":velocity_magnitude} # return a dict!


def calc_ll(v_0, air_density, tsr, airfoil, radius, n_blades, inner_radius,
            pitch, resolution_ll, vortex_core_radius, debug,
            disctretization="sin",
            residual_max=1e-10, n_iter_max=1000):
    
    #--------------Get twist and chord distributions ----------------#
    
    ##!!!!!!! Needs to be adapted!
    # Get the positions along the span for each element and the nodes
    # Compute discretization of the blade:
    # uniform distribution or cosine distribution

    if disctretization =="uniform":
        radii_ends = np.linspace(inner_radius, radius, resolution_ll) # radial positions of the ends of each section (uniform)
        
    else:
        radii_ends = (np.sin(np.linspace(-np.pi/2, np.pi/2, resolution_ll))/2+0.5)*(radius-inner_radius)+inner_radius # sine
    # distribution

    # M: changed chord and twist to the edges of an element.... if that really makes sense needs to be discussed
    # J: It absolutely does, the confusion probably roots in some misunderstanding. In contrast to BEM, the definition
    # of the blade for our lifting line is first and foremost solely to define the geometry of the vortex system for
    # which we cannot use anything but the ends of the blade elements (because there the trailing vortices start).

    chord_ends = twist_chord.get_chord(radii_ends, radius) # chord at the ends of each section
    twist_ends = -twist_chord.get_twist(radii_ends, radius)
    pitch = -pitch
    # check /documentation/VortexSystem.pdf for the angles to understand the signs. In short: the conventional twist
    # and pitch definitions face the opposite direction of the angle of the vortex_system coordinate system that is
    # responsible for these rotations (angle: blade_rotation). Thus, the "-" "transforms" the conventional blade
    # coordinate system to the coordinate system of vortex_system.

    # Now we have the geometry of the blade. This knowledge is inserted into a vortex_system object in PART 2.


    # -------------- Run BEM ----------------#
    # We now want to run BEM to obtain a CT value which we can use to compute the wake convection
    
    induction, bem_results = calc_induction_bem(tsr, -2)
    u_rotor = v_0*(1-induction)
    print("BEM done")
    
    #------------------------------------------------------#
    # PART 2 - set up wake system
    #------------------------------------------------------#
    
    #------------- Compute the wake structure ----------#
    
    # compute inputs for the wake tool:
    omega = tsr*v_0/radius
    wake_speed = u_rotor  # take the speed at the rotor or the speed far downstream? Probably at the rotor is a closer guess
    wake_length = 1 * 2*radius # how many diameters should the wake have?
    resolution_wake = 50

    # initializy wake geometry
    vortex_system = VortexSystem()
    vortex_system.set_blade(radii_ends, chord_ends, blade_rotation=twist_ends+pitch, rotor_rotation_speed=omega,
                            n_blades=n_blades)

    # set the properties of the wake.
    # M: note that the resolution here should be something related to the discretisation of the trailing vortices
    # rather than the discretisation of the blade
    # J: The resolution is purely related to the length of the trailing vortices and stands in no relation to the
    # discretisation of the blade, that is the resolution is the number of points+1 per trailing vortex.
    vortex_system.set_wake(wake_speed=wake_speed, wake_length=wake_length, resolution=resolution_wake)

    vortex_system.rotor() # create the vortex system, that is bound and trailing vortices of the whole rotor
    #------------------------------------------------------#
    # PART 3 - Compute matrices
    #------------------------------------------------------#
    # 3.1 Set Control points
    print("Create vortex system")
    # The control points have to lie on the quarter chord. We assume them to be radially in the middle of each blade
    # element
    vortex_system.set_control_points_on_quarter_chord()

    # 3.2 Create the matrices connecting the circulation and the velocity field
    print("Create induction matrices")
    # calculate the trailing induction matrices
    trailing_mat_u, trailing_mat_v, trailing_mat_w = vortex_system.trailing_induction_matrices(vortex_core_radius=vortex_core_radius)
    bound_mat_u, bound_mat_v, bound_mat_w = vortex_system.bound_induction_matrices(vortex_core_radius=vortex_core_radius) # calculate the bound induction matrices
    if debug:
        vortex_system.blade_elementwise_visualisation(control_points=True)
        # vortex_system.rotor_visualisation(control_points=True)
    #------------------------------------------------------#
    # PART 4 - Iterate to find the right circulation
    #------------------------------------------------------#
    
    print("Initialize values")
    # 4.1 Initialize the arrays for the velocity
    radii_centre = (radii_ends[:-1]+radii_ends[1:])/2 # radial position of the centre of each blade element
    twist_centre = -twist_chord.get_twist(radii_centre, radius) # twist at the centre of each element
    chord_centre = twist_chord.get_chord(radii_centre, radius) # chord at the centre of each element

    #u_induced = np.zeros(radii_centre.shape)
    # make interpolation function of the induction 

    u_induced_function = scipy.interpolate.interp1d(bem_results.r_centre, bem_results.a)
    v_induced_function = scipy.interpolate.interp1d(bem_results.r_centre, bem_results.a_prime)
    u_induced = v_0 * u_induced_function(radii_centre)
    v_induced = omega * radii_centre * v_induced_function(radii_centre)
    #u_induced = np.zeros(radii_centre.shape)
    #v_induced = np.zeros(radii_centre.shape)
    inflow_velocity = calc_velocity(v_0, omega, radii_centre, u_induced, v_induced) # we now have the u,v velocity
    # vector
    # We can compute the effective angle of attack with it
    # M: -> This needs to be changed -> shouldnt use induced velocity for that
    # J: can we delete some of these lines for clarity?
    # effective_aoa = twist_list_centre + np.rad2deg(np.arctan(-v_induced/u_induced)) # second part should be 0 here,
    # as were not yet inducing velocities
    #effective_aoa = twist_list_centre + np.rad2deg(np.arctan(-inflow_velocity["v"]/inflow_velocity["u"])) # second
    # part should be 0 here, as were not yet inducing velocities
    #effective_aoa = twist_list_centre + (np.arctan(-inflow_velocity["v"]/inflow_velocity["u"])) # second part
    # should be 0 here, as were not yet inducing velocities -----> in radian

    # again, check /documentation/VortexSystem.pdf for the angles to understand the signs and tan use.
    effective_aoa = np.arctan(inflow_velocity["u"]/inflow_velocity["v"])+(pitch+twist_centre) # in rad,
    # the + is because the pitch and twist are defined in the vortex_system coordinate system.

    lift, cl_current, cd_current = calc_lift(effective_aoa, chord_centre, inflow_velocity["magnitude"])

    # from the lift we can obtain the bound circulation
    bound_circulation = calc_circulation(lift,
                                         inflow_velocity["magnitude"],
                                         lift/(air_density*inflow_velocity["magnitude"])) # M: compute with the
    # magnitude -> is that so exact ?
    # J: Yes that's perfect.

    # The trailing circulation is the difference between two bound circulations, or the "step"
    trailing_circulation = np.append(0, bound_circulation)-np.append(bound_circulation, 0)
    # Everything is defined, now create a loop to iterate over the circulations
    residual = 1
    n_iter = 0
    L_mag_new = 0
    while residual > residual_max and n_iter<=n_iter_max:
        L_mag = L_mag_new # magnitude of the lift
        u_induced = trailing_mat_u @ trailing_circulation + bound_mat_u @ bound_circulation # calculate stream-wise induction
        v_induced = trailing_mat_v @ trailing_circulation + bound_mat_v @ bound_circulation # calc azimuthal velocity / downwash

        inflow_velocity = calc_velocity(v_0, omega, radii_centre, u_induced, v_induced) # we now have the u,v velocity
        # vector
        effective_aoa = np.arctan(inflow_velocity["u"]/inflow_velocity["v"])+(pitch+twist_centre)

        lift, cl_current, cd_current = calc_lift(effective_aoa, chord_centre, inflow_velocity["magnitude"])

        # from the lift we can obtain the bound circulation
        bound_circulation = calc_circulation(lift, inflow_velocity["magnitude"], bound_circulation)

        trailing_circulation = np.append(0, bound_circulation)-np.append(bound_circulation, 0)
        #trailing_circulation = np.reshape(trailing_circulation, (radii.size,1))
        
        L_mag_new = np.sum(lift)
        residual = np.abs(L_mag-L_mag_new)
        # update circulation
        if n_iter%20==0:
            print(f"Iter: {n_iter} \t residual: {residual}")
        n_iter +=1

    #----------------- Compute lift coefficient -----#
    cl = lift / (0.5 * air_density * inflow_velocity["magnitude"] ** 2 * chord_centre)  # lift coefficient
    axial_induction = - u_induced / v_0 
    a_prime = v_induced / (omega * radii_centre)  

    ll_results = {"r_centre": radii_centre, "u_induced": u_induced, "lift": lift,
                  "cl": cl, "cd": cd_current, "a": axial_induction, 
                  "a_prime": a_prime, "aoa":np.rad2deg(effective_aoa),
                  "bound_circulation":bound_circulation,
                  "bem_results": bem_results}

    return ll_results

def task1(debug=False):
    """
    Function to combine all operations in Task 1 and returns the data as a dictionary

    That is:
    - computing the induction via bem
    - set up the wake accordingly
    - create the matrix system for the geometrical induction via the vortices
    - solve for circulation iteratively
    
    step 4 needs debugging
    """
    
    #------------ Get all the inputs -------------#
    radius = 50                         # radius of the rotor
    n_blades = 3                        # number of blades
    inner_radius = 0.2 * radius         # inner end of the blade section
    pitch_deg = -2                      # pitch in degrees
    pitch = np.deg2rad(pitch_deg)       # pitch angle in radian
    resolution_ll = 14                     # Spanwise resolution -> seems to break for larger values
    residual_max = 1e-10
    n_iter_max = 1000
    vortex_core_radius = 1
    
    #------------ Operational data ---------------#
    v_0 = 10                            # [m] Wind speed
    air_density = 1.225
    tsr = 8                      # Tip speed ratios  to be calculated
    airfoil = pd.read_excel("../data/polar.xlsx", skiprows=3)    # read in the airfoil. Columns [alpha, cl, cd cm]
    
    

    ll_results = calc_ll(v_0, air_density, tsr, airfoil, radius, n_blades,
                         inner_radius, pitch, resolution_ll, vortex_core_radius,
                         debug, disctretization="uniform",
                         residual_max=residual_max, n_iter_max=n_iter_max)

    #---------------- Plotting ----------------------#
    if debug:
        fig, axs = plt.subplots(7, 1)
        axs[0].plot(radii_centre, bound_circulation, "x")
        axs[1].plot(radii_ends, trailing_circulation)
        axs[2].plot(radii_centre, cl)
        axs[3].plot(radii_centre, u_induced)
        axs[4].plot(radii_centre, v_induced)
        axs[5].plot(radii_centre, u_induced/v_0)
        axs[6].plot(radii_centre, v_induced / (omega * radii_centre))
        axs[2].plot(bem_results.r_centre, bem_results.c_l)
        helper.handle_axis(axs, x_label="radial position", grid=True, line_width=3,
                           font_size=12, y_label=["bound\ncirculation", "trailing\ncirculation",
                                    "Cl", "u induced", "v induced", "induction", "a'"])
        helper.handle_figure(fig, show=True, size=(6, 12))
        
        plt.figure(2)
        plt.plot(radii_centre, cl)
        plt.plot(bem_results.r_centre, bem_results.c_l)
        plt.ylim([0,1.5])
        plt.show()
    return ll_results

def compare_ll_bem(v_0: float):
    """
    Function to compare the lifting line and the BEM results.
    This function is intended to be purely plotting
    """
    results = task1(debug=False)

    fig, axs = plt.subplots(6, 1)
    # CL
    axs[0].plot(results["r_centre"], results["cl"])
    axs[0].plot(results["bem_results"]["r_centre"], results["bem_results"]["c_l"])
    # Cd
    axs[1].plot(results["r_centre"], results["cd"])
    axs[1].plot(results["bem_results"]["r_centre"], results["bem_results"]["c_d"])
    # induction 
    axs[2].plot(results["r_centre"], results["a"])
    axs[2].plot(results["bem_results"]["r_centre"], results["bem_results"]["a"])
    
    # a prime
    axs[3].plot(results["r_centre"], results["a_prime"])
    axs[3].plot(results["bem_results"]["r_centre"], results["bem_results"]["a_prime"])
    
    #aoa 
    axs[4].plot(results["r_centre"], results["aoa"])
    axs[4].plot(results["bem_results"]["r_centre"], results["bem_results"]["alpha"])
    
    # circulation
    axs[5].plot(results["r_centre"], results["bound_circulation"])
    axs[5].plot(results["bem_results"]["r_centre"], results["bem_results"]["circulation"])

    # make plots look nice
    helper.handle_axis(axs, x_label="radial position", grid=True, line_width=3,
                       font_size=12, y_label=["Cl", "Cd", "a", 
                                              "a_prime", "Angle of \nattack",
                                              "circulation"])
    helper.handle_figure(fig, show=True, size=(6, 12))
    #plt.show()

if __name__== "__main__":
    v_0 = 10
    compare_ll_bem(v_0)
