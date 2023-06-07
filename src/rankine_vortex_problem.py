from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import Helper
helper = Helper()


def vortex_induction_factor(vortex_start: np.ndarray,
							vortex_end: np.ndarray,
							induction_point: np.ndarray,):
	"""
	This function calculates the induction at a point 'induction_point' from a straight vortex line between the
	two points 'vortex_start' and 'vortex_end' for a unity circulation. The returned value is a vector of induced
	velocities.
	:param vortex_start: numpy array of size (3,)
	:param vortex_end: numpy array of size (3,)
	:param induction_point: numpy array of size (3,)
	:param core_radius: radius of the solid body inside the vortex
	:return: numpy array os size (3,)
	"""
	r_s = induction_point-vortex_start  # vector from the start of the vortex to the induction point
	r_e = induction_point-vortex_end  # vector from the end of the vortex to the induction point
	r_v = vortex_end-vortex_start  # vector from the start of the vortex to the end of the vortex

	l_s = np.linalg.norm(r_s)  # distance between the induction point and the start of the vortex
	l_e = np.linalg.norm(r_e)  # distance between the induction point and the end of the vortex
	l_v = np.linalg.norm(r_v)  # length of the vortex

	h = np.linalg.norm(np.cross(r_v, r_s))/l_v  # shortest distance between the control point and an infinite
	# extension of the vortex filament
	e_i = np.cross(r_v, r_s)/(h*l_v)  # unit vector of the direction of induced velocity
	return 1/(2*np.pi*h)*e_i, e_i/(4*np.pi*h*l_v)*(np.dot(r_v, (r_s/l_s-r_e/l_e)))


fig, ax = plt.subplots()
v_lengths = [0.1, 0.5, 1, 2, 10]


for i, v_length in enumerate(v_lengths):
	v_start = np.asarray([10, 0, -v_length/2])
	v_end = np.asarray([10, 0, v_length/2])
	x_points = np.linspace(0, 9, 50)
	vortex_inductions = list()
	rankine_inductions = list()
	for x in x_points:
		cp = np.asarray([x, 0, 0])
		rankine_induction, vortex_induction = vortex_induction_factor(v_start, v_end, cp)
		rankine_inductions.append(abs(rankine_induction[1]))
		if i == 0:
			vortex_inductions.append(abs(vortex_induction[1]))

	if i == 0:
	ax.plot(x_points, rankine_inductions[::-1], label="solid body")
	ax.plot(x_points, vortex_inductions[::-1], label="irrotational vortex")

helper.handle_axis(ax, grid=True, legend=True)


