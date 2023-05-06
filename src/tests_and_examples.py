import numpy as np
from geometry import FrozenWake

test = {
    "wake_visualisation": False,
    "induction_matrix": True
}

if test["wake_visualisation"]:
    wake = FrozenWake()
    wake.set_rotor(0.2+np.linspace(0,1,5), np.linspace(0,0.2,5)[::-1], blade_rotation=-0.5*np.pi/2,
                   rotor_rotation_speed=np.pi/4)
    wake.set_wake_properties(wake_speed=0.5, wake_length=5, time_resolution=50)
    wake.rotor()
    wake.blade_elementwise_visualisation()
    # wake.rotor_visualisation()


if test["induction_matrix"]:
    wake = FrozenWake()
    wake.set_rotor(0.2+np.linspace(0,1,4), 1, blade_rotation=-0*np.pi/2,
                   rotor_rotation_speed=0, n_blades=3)
    wake.set_wake_properties(wake_speed=0.5, wake_length=5, time_resolution=5)
    wake.rotor()
    induction_matrices, control_points = wake.induction_matrices(0, 0, 0.2+np.linspace(0,1,4))
    print("x", "\n", induction_matrices[0])
    print("y", "\n", induction_matrices[1])
    print("z", "\n", induction_matrices[2])
    wake.rotor_visualisation(control_points)
