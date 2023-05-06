import numpy as np
from geometry import VortexSystem

test = {
    "wake_visualisation": True,
    "induction_matrix": False
}

if test["wake_visualisation"]:
    vortex_system = VortexSystem()
    vortex_system.set_blade(0.2+np.linspace(0,1,5), np.linspace(0, 0.2, 5)[::-1], blade_rotation=-0 * np.pi / 2,
                            rotor_rotation_speed=np.pi / 4)
    vortex_system.set_wake_properties(wake_speed=0.5, wake_length=5, time_resolution=50)
    # vortex_system.blade()
    vortex_system.rotor()
    vortex_system.set_control_points(x_control_points=0,
                                     y_control_points=0.25*np.linspace(0, 0.2, 5)[::-1],
                                     z_control_points=0.2+np.linspace(0,1,5))
    vortex_system.blade_elementwise_visualisation(control_points=True)
    vortex_system.rotor_visualisation(control_points=True)


if test["induction_matrix"]:
    vortex_system = VortexSystem()
    vortex_system.set_blade(0.2 + np.linspace(0, 1, 4), 0.3, blade_rotation=-0 * np.pi / 2, rotor_rotation_speed=0.8 * np.pi,
                   n_blades=3)
    vortex_system.set_wake_properties(wake_speed=0.5, wake_length=1, time_resolution=20)
    vortex_system.rotor()
    vortex_system.set_control_points(x_control_points=0, y_control_points=0, z_control_points=np.linspace(0,1, 5))
    ind_u, ind_v, ind_w= vortex_system.trailing_induction_matrices()
    print("x", "\n", ind_u)
    print("y", "\n", ind_v)
    print("z", "\n", ind_w)
    vortex_system.rotor_visualisation()
