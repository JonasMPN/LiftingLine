import numpy as np
from vortex_system import VortexSystem
import matplotlib.pyplot as plt
from helper_functions import Helper
helper = Helper()

test = {
    "wake_visualisation": False,
    "induction_matrix": True,
    "lifting_line": True
}

if test["wake_visualisation"]:
    vortex_system = VortexSystem()
    vortex_system.set_blade(0.2+np.linspace(0,1,5), np.linspace(0, 0.2, 5)[::-1], blade_rotation=-0*np.pi/2,
                            rotor_rotation_speed=np.pi/4, n_blades=3)
    vortex_system.set_wake(wake_speed=0.5, wake_length=5, resolution=50)
    # vortex_system.blade()
    vortex_system.rotor()
    vortex_system.set_control_points(x_control_points=0,
                                     y_control_points=0.25*np.linspace(0, 0.2, 5)[::-1],
                                     z_control_points=0.2+np.linspace(0,1,5))
    vortex_system.blade_elementwise_visualisation(control_points=True)
    vortex_system.rotor_visualisation(control_points=True)


if test["induction_matrix"]:
    vortex_system = VortexSystem()
    vortex_system.set_blade(0.2+np.linspace(0,1,5), 1, blade_rotation=-0*np.pi/2,
                            rotor_rotation_speed=0*np.pi/4, n_blades=1)
    vortex_system.set_wake(wake_speed=0.5, wake_length=5, resolution=3)
    # vortex_system.blade()
    vortex_system.rotor()
    vortex_system.set_control_points(x_control_points=0.25,
                                     y_control_points=0,
                                     z_control_points=0.2+np.linspace(0,1,5))
    t_ind_u, t_ind_v, t_ind_w = vortex_system.trailing_induction_matrices()
    print("Trailing induction")
    print("x", "\n", t_ind_u)
    print("y", "\n", t_ind_v)
    print("z", "\n", t_ind_w)
    # b_ind_u, b_ind_v, b_ind_w = vortex_system.bound_induction_matrices()
    # print("Bound induction")
    # print("x", "\n", b_ind_u)
    # print("y", "\n", b_ind_v)
    # print("z", "\n", b_ind_w)
    vortex_system.rotor_visualisation(control_points=True)


if test["lifting_line"]:
    # tests the lifting line model for a rectangular flat plate. Assumed density=1 and chord=1.
    resolution = 20 # number of shed trailing vortex lines
    wake_length = 20
    aspect_ratio = 34 # aspect ratio of the flat plate. Influences the number of control points (by influencing the
    # number of plate elements)
    inflow_speed = 10 # inflow speed
    aoa = 5 # angle of attack in degree
    r = np.arccos(np.linspace(0,1,int(aspect_ratio/4)))/(np.pi/2)
    r = (np.append(-r[:-1], r[::-1])+1)*aspect_ratio/2 # plate elements from 0 to chord*aspect_ratio (here =
    # aspect_ratio)

    vortex_system = VortexSystem()
    vortex_system.set_blade(r_elements=r, c_elements=1, blade_rotation=np.deg2rad(aoa)-np.pi/2, rotor_rotation_speed=0,
                            n_blades=1) # define a flat plate by constant chord length, no rotational speed and only
    # one blade
    vortex_system.set_wake(wake_speed=inflow_speed, wake_length=wake_length, resolution=resolution)
    vortex_system.rotor() # calculate the vortex system of the rotor (meaning bound and trailing vortices)
    # now set the control points on the quarter chord of the plate in the middle of each plate element
    vortex_system.set_control_points(x_control_points=1/4*np.cos(np.deg2rad(aoa)),
                                     y_control_points=1/4*np.sin(np.deg2rad(aoa)),
                                     z_control_points=1/2*(r[:-1]+r[1:]))

    # fig, ax = vortex_system.rotor_visualisation(control_points=True, show=False) # plot vortex system
    # helper.handle_axis(ax, x_label="x", y_label="y", y_lim=(0,1), grid=True) # visual modifications
    # plt.show() # uncomment above three lines to show the plot

    t_ind_u, t_ind_v, t_ind_w = vortex_system.trailing_induction_matrices() # calculate the trailing induction matrices
    b_ind_u, b_ind_v, b_ind_w = vortex_system.bound_induction_matrices() # calculate the bound induction matrices

    # flat plate
    def lift(aoa, inflow_mag): return np.pi*inflow_mag**2*np.sin(np.deg2rad(aoa)) # mind density=1, chord=1
    def circulation(lift, inflow_speed): return lift/inflow_speed

    U_inflow = np.ones((r.size-1,1))*inflow_speed # inflow velocity at control points
    L = lift(aoa, U_inflow) # lift at control points
    L_mag = np.sum(L) # lift of flat plate
    bound_circulation = circulation(L, inflow_speed)
    trailing_circulation = np.append(0, bound_circulation)-np.append(bound_circulation, 0) # the trailing circulation
    # between two plate elements arises from the difference in circulation of the bound vortices of those elements.
    trailing_circulation = np.reshape(trailing_circulation, (r.size, 1))
    L_mag_new = 0

    while np.abs(L_mag-L_mag_new) > 1e-8:
        L_mag = L_mag_new
        u_ind = t_ind_u@trailing_circulation + b_ind_u@bound_circulation # calculate stream-wise induction
        V = t_ind_v@trailing_circulation + b_ind_v@bound_circulation # calculate downwash
        U = U_inflow+u_ind # resulting stream-wise velocity
        inflow_speed = np.sqrt(U*U+V*V) # new velocity magnitude per control point
        eff_aoa = aoa + np.rad2deg(np.arctan(-V/U)) # new effective angle of attack per control point
        L = lift(eff_aoa, inflow_speed) # new lift per control point
        L_mag_new = np.sum(L)

        bound_circulation = circulation(L, inflow_speed) # new bound circulation
        trailing_circulation = np.append(0, bound_circulation)-np.append(bound_circulation, 0)
        trailing_circulation = np.reshape(trailing_circulation, (r.size, 1)) # new trailing vortices


    # some plotting
    fig, ax = plt.subplots()
    ax.plot(1/2*(r[:-1]+r[1:])/aspect_ratio, L/(1/2*inflow_speed**2))
    helper.handle_axis(ax, x_label="span/(c*AR)", y_label=r"$c_l$", grid=True, font_size=25,
                       line_width=5, y_lim=(0, 0.55))
    helper.handle_figure(fig, show=True)


