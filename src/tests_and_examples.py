import numpy as np
from geometry import FrozenWake

test = {
    "wake": True
}

if test["wake"]:
    wake = FrozenWake()
    wake.set_rotor(0.2+np.linspace(0,1,5), np.linspace(0,0.2,5)[::-1], blade_rotation=0*np.pi/2,
                   rotor_rotation_speed=np.pi/4)
    wake.rotor(wake_speed=0.5, wake_length=5, time_resolution=50)
    wake.blade_elementwise_visualisation()
    # wake.rotor_visualisation()