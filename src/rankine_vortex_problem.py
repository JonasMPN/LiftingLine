from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import Helper
helper = Helper()

do = {
	"r_change": False,
	"v_jump": False,
	"v_comp": True
}


def vortex_induction_factor_1(vortex_start: np.ndarray,
							 vortex_end: np.ndarray,
							 induction_point: np.ndarray):
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

if do["r_change"]:
	res = 100
	max_distance = 3
	min_distance = 0.3
	fig, ax = plt.subplots()
	v_lengths = [0.1, 0.5, 1, 2, 5]
	for i, v_length in enumerate(v_lengths):
		v_start = np.asarray([0, 0, -v_length/2])
		v_end = np.asarray([0, 0, v_length/2])
		x_points = np.linspace(min_distance, max_distance, res)[::-1]
		vortex_inductions = list()
		solid_body = list()
		for x in x_points:
			cp = np.asarray([x, 0, 0])
			solid_body_ind, vortex_induction = vortex_induction_factor_1(v_start, v_end, cp)
			vortex_inductions.append(abs(vortex_induction[1]))
			if i == 0:
				solid_body.append(abs(solid_body_ind[1]))
		if i == 0:
			ax.plot(x_points, solid_body, "--k", label="solid body")
		ax.plot(x_points, vortex_inductions, label=f"irrotational vortex, l={v_length}")
	
	helper.handle_axis(ax, grid=True, legend=True, line_width=3, font_size=23, x_label="radial distance (m)",
					   y_label=r"$U_{ind}/\Gamma$ (1/m)", title="Induced velocities")
	helper.handle_figure(fig, save_to="../results/rankine_vortex/r_change.png", size=(10, 8))

if do["v_jump"]:
	fig, ax = plt.subplots()
	core_radius = 1
	min_length = 0.1
	max_length = 8
	v_lengths = np.linspace(min_length, max_length, res)
	induction_point = np.asarray([core_radius, 0, 0])
	differences = list()
	for v_length in v_lengths:
		v_start = np.asarray([0, 0, -v_length/2])
		v_end = np.asarray([0, 0, v_length/2])
		solid_body_ind, vortex_induction = vortex_induction_factor_1(v_start, v_end, induction_point)
		differences.append(solid_body_ind[1]-vortex_induction[1])
	ax.plot(v_lengths, differences)
	helper.handle_axis(ax, grid=True, line_width=3, font_size=23, x_label="vortex length (m)",
					   y_label=r"$\Delta U_{ind}/\Gamma$ (1/m)", title=r"Velocity jump at $r=1$m")
	helper.handle_figure(fig, save_to=f"../results/rankine_vortex/v_jump_r{core_radius}.png", size=(10, 8))

def vortex_induction_factor_2(vortex_start: np.ndarray,
							  vortex_end: np.ndarray,
							  induction_point: np.ndarray,
							  core_radius: float):
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
	return h/(2*np.pi*core_radius**2)*e_i, e_i/(4*np.pi*h*l_v)*(np.dot(r_v, (r_s/l_s-r_e/l_e)))


if do["v_comp"]:
	def velocity(radius_core: float, circulation: float, radii: np.ndarray) -> np.ndarray:
		solid_body_radii = radii[radii < radius_core]
		irrotational_radii = radii[radii >= radius_core]
		solid_body_rot_vel = circulation/(2*np.pi*core_radius**2)
		return np.append(solid_body_rot_vel*solid_body_radii, circulation/(2*np.pi*irrotational_radii))
	
	
	radii = np.linspace(0, 6, 1000)
	circulations = [1, 2, 5]
	core_radii = [0.5, 1, 3]
	fig, ax = plt.subplots()
	vel = {circ: list() for circ in circulations}
	v_length = 3
	v_start = np.asarray([0, 0, -v_length/2])
	v_end = np.asarray([0, 0, v_length/2])
	for circulation, core_radius in zip(circulations, core_radii):
		ax.plot(radii, velocity(core_radius, circulation, radii), label=rf"$\Gamma={circulation}$, $r_c={core_radius}$")
		for r in radii:
			induction_point = np.asarray([r, 0, 0])
			solid_body_ind, vortex_induction = vortex_induction_factor_2(v_start, v_end, induction_point, core_radius=core_radius)
			if r < core_radius:
				vel[circulation].append(solid_body_ind[1]*circulation)
			else:
				vel[circulation].append(vortex_induction[1]*circulation)
	helper.handle_axis(ax, grid=True, legend=True, line_width=3, font_size=23, x_label="radial position (m)",
					   y_label=r"$U_{ind}$ (m/s)", title="Azimuthal velocities around different Rankine vortices")
	helper.handle_figure(fig, save_to=f"../results/rankine_vortex/rankine_vortex.png", size=(10, 8))
	
	fig, ax = plt.subplots()
	labels = {1: r"$\Gamma=1$, $r_c=0.5$",
			  2: r"$\Gamma=2$, $r_c=1$",
			  5: r"$\Gamma=5$, $r_c=2$"}
	for circ, velocities in vel.items():
		ax.plot(radii, vel[circ], label=labels[circ])
	helper.handle_axis(ax, grid=True, legend=True, line_width=3, font_size=23, x_label="radial position (m)",
					   y_label=r"$U_{ind}$ (m/s)", title="Jump in velocity for different 'Rankine' vortices")
	helper.handle_figure(fig, save_to=f"../results/rankine_vortex/v_comp.png", size=(10, 8))


