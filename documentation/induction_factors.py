import numpy as np

def vortex_induction_factor(vortex_start: np.ndarray, vortex_end: np.ndarray, induction_point: np.ndarray) -> np.ndarray:
	"""
	This function calculates the induction at a point 'induction_point' from a straight vortex line between the
	two points 'vortex_start' and 'vortex_end' for a unity circulation. The returned value is a vector of induced
	velocities.
	:param vortex_start: numpy array of size (3,)
	:param vortex_end: numpy array of size (3,)
	:param induction_point: numpy array of size (3,)
	:return: numpy array os size (3,)
	"""
	r_s = induction_point-vortex_start # vector from the start of the vortex to the induction point
	r_e = induction_point-vortex_end # vector from the end of the vortex to the induction point
	r_v = vortex_end-vortex_start # vector from the start of the vortex to the end of the vortex

	l_s = np.linalg.norm(r_s) # distance between the induction point and the start of the vortex
	l_e = np.linalg.norm(r_e) # distance between the induction point and the end of the vortex
	l_v = np.linalg.norm(r_v) # length of the vortex

	h = np.linalg.norm(np.cross(r_v, r_s))/l_v # shortest distance between the control point and an infinite
	# extension of the vortex filament
	if h <= 1e-10: # the control point lies too close (for stability reasons) normal to the vortex line
		# todo handle control points that lie very close to the vortex core with a solid body rotation
		return np.zeros(3) # for now assume there is no induction
	e_i = np.cross(r_v, r_s)/(h*l_v) # unit vector of the direction of induced velocity
	return  e_i/(4*np.pi*h*l_v)*(np.dot(r_v, (r_s/l_s-r_e/l_e)))