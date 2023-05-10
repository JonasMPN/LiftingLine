import numpy as np
def _vortex_induction_factor(vortex_start: np.ndarray,
                             vortex_end: np.ndarray,
                             induction_point: np.ndarray) -> np.ndarray:
    """
    This function calculates the induction at a point 'induction_point' from a straight vortex line between the
    two points 'vortex_start' and 'vortex_end' for a unity circulation. The returned value is a vector of induced
    velocities.
    :param vortex_start:
    :param vortex_end:
    :param induction_point:
    :return:
    """
    vec_R_1 = induction_point - vortex_start  # vector from vortex_start to the induction point
    vec_R_2 = induction_point - vortex_end  # vector from vortex_end to the induction point
    R_1 = np.linalg.norm(vec_R_1)  # distance between the vortex start point and the induction point
    R_2 = np.linalg.norm(vec_R_2)  # distance between the vortex end point and the induction point
    vec_plane_normal = np.cross(vec_R_1, vec_R_2)  # vector that's normal on the plane spanned by the three points
    if np.linalg.norm(vec_plane_normal) <= 1e-10:  # this happens when the induction point lies on the extended
        # vortex line or very close around it
        # todo handle control points that lie very close to the vortex core
        return np.zeros(3)
    l_sq_plane_normal = np.dot(vec_plane_normal, vec_plane_normal)  # squared length of that plane normal vector
    vec_vortex = vortex_end - vortex_start  # vector representing the vortex line
    fac_1 = np.dot(vec_vortex, vec_R_1)  # zero clue what this does later
    fac_2 = np.dot(vec_vortex, vec_R_2)  # zero clues what this does later
    K = 1 / (4 * np.pi * l_sq_plane_normal) * (fac_1 / R_1 - fac_2 / R_2)  # some magic factor
    return K * vec_plane_normal  # boom done