# functions to obtain twist and chord from BEM assignment

import numpy as np
def get_twist(r: float or np.ndarray, r_max: float) -> float or np.ndarray:
    """
        function to get the twist along the blade in radians
        r: radial position along the blade in [m]
        r_max: maximum radius of the blade in [m]
        out: twist in radians
    """
    return np.radians(14*(1-r/r_max))

def get_chord(r: float or np.ndarray, r_max: float) -> float or np.ndarray:
    """
        function to calculate chord along the span in m
        r: radial position along the blade in [m]
        r_max: maximum radius of the blade in [m]
    """
    return 3*(1-r/r_max)+1