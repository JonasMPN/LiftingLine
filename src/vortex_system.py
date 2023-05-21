# class to create the vortex system of a wing, a blade or a whole rotor
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


class VortexSystem:
    def __init__(self):
        """
        Initialise VortexSystem object.
        """
        # blade properties
        self.r_elements = None
        self.c_elements = None
        self.blade_rotation = None

        # rotor properties
        self.rotor_rotation_speed = None
        self.n_blades = None

        # wake properties
        self.wake_speed = None
        self.wake_length = None
        self.resolution = None

        # rotor array properties
        self.rotor_positions = None
        self.rotor_rotations = None

        # control points
        self.control_points = None
        self.n_control_points = None

        # instance properties
        # bound vortices. The 'X-wise' ones have a structure that allow for foolproof access of which 'X' the
        # coordinates come from. The same variable without the 'X-wise' has all coordinates combined in one large
        # numpy array.
        self.coordinates_blade_bound_elementwise = None
        self.coordinates_blade_bound = None

        self.coordinates_rotor_bound_bladewise = None
        self.coordinates_rotor_bound = None

        self.coordinates_rotor_array_bound_rotorwise = None
        self.coordinates_rotor_array_bound = None

        # trailing vortices
        self.coordinates_blade_trailing_elementwise = None
        self.coordinates_blade_trailing = None

        self.coordinates_rotor_trailing_bladewise = None
        self.coordinates_rotor_trailing = None

        self.coordinates_rotor_array_trailing_rotorwise = None
        self.coordinates_rotor_array_trailing = None

        # error clarification
        self.rotor_set = False
        self.rotor_array_set = False
        self.wake_set = False
        self.blade_calculated = False
        self.rotor_calculated = False
        self.rotor_array_calculated = False

    def set_blade(self,
                  r_elements: np.ndarray,
                  c_elements: float or np.ndarray,
                  blade_rotation: float or np.ndarray,
                  rotor_rotation_speed: float,
                  n_blades: int = 3
                  ) -> None:
        """
        The rotor coordinate system is such that the x faces downwind along the rotor axis, y faces to the left
        facing the rotor from the front (i.e. looking downwind) and z point upwards. z goes along the leading edge of a
        blade that is point upward. All angles follow a standard right-hand rule. See /documentation/VortexSystem.pdf
        for details.
        In the following definitions, 'section' means the blade profile at a certain location. A blade element is
        defined by two blade sections, i.e., one blade element is bound by two blade sections (beginning and end).

        :param r_elements:              blade section radial position
        :param c_elements:              blade section chord length 
        :param blade_rotation:          rotation of the blade sections around the z-axis in radians. An angle of 0
        means the chord is parallel to the y-axis with the trailing edge pointing in the positive y-direction.
        :param rotor_rotation_speed:    Rotational speed of the rotor
        :param n_blades:                Number of blades
        :return:                        None
        """
        self.r_elements = r_elements
        # checks if a list of chord and blade rotations is given. If not, create a np.ndarray with the uniform value
        # of the float using the _float_to_ndarray function
        self.c_elements, self.blade_rotation = self._float_to_ndarray(len(r_elements), c_elements, blade_rotation)
        # Ensure that r_elements, c_elements, and blade_rotation expect the same number of sections
        if self.r_elements.shape != self.c_elements.shape or self.r_elements.shape  != self.blade_rotation.shape:
            raise ValueError("'r_elements', 'c_elements', and 'blade_rotation' do not have the same length. This only "
                             "happens if 'c_elements' or 'blade_rotation' are specified as a list. The shapes are:"
                             f"r_elements: {self.r_elements.size}, c_elements: {self.c_elements.size}, blade_rotation:"
                             f" {self.blade_rotation.size}")
        self.rotor_rotation_speed = rotor_rotation_speed
        self.n_blades = n_blades

        # if a new rotor is defined, the old vortex system is no longer correct and is therefore deleted
        # bound vortex system
        self.coordinates_blade_bound_elementwise = None
        self.coordinates_blade_bound = None
        self.coordinates_rotor_bound_bladewise = None
        self.coordinates_rotor_bound = None
        self.coordinates_rotor_array_bound_rotorwise = None
        self.coordinates_rotor_array_bound = None

        # trailing vortices
        self.coordinates_blade_trailing_elementwise = None
        self.coordinates_blade_trailing = None
        self.coordinates_rotor_trailing_bladewise = None
        self.coordinates_rotor_trailing = None
        self.coordinates_rotor_array_trailing_rotorwise = None
        self.coordinates_rotor_array_trailing = None

        # 'rotor_set' is internally used to provide meaningful error messages if the user wants to perform
        # calculations that require the rotor properties to be set.
        self.rotor_set = True
        return None

    def set_rotor_array(self,
                        rotor_positions: list[list] or list[np.ndarray],
                        rotor_rotations: list or np.ndarray) -> None:
        """
        Sets the parameters of the rotor array. 'Rotor array' is an umbrella term for wind farms if the rotors are
        wind turbines, and for distributed propulsion if they are aircraft propellers. The number of elements the
        lists of 'rotor_positions' and 'rotor_rotations' have is the number of rotors that are created.
        :param rotor_positions: A list of points at which the rotor centre of each rotor is. Each point has three
                                coordinates.
        :param rotor_rotations: A list of the rotor rotations which define the position of the blades with respect to
                                the rotor axes. Check /documentation/VortexSystem.pdf for the angle definitions.
        :return:
        """
        if len(rotor_positions) != len(rotor_rotations):
            raise ValueError(f"'rotor_positions' suggest a different number of rotors ({len(rotor_positions)}) than "
                             f"'rotor_rotations' does ({len(rotor_rotations)}). Both must be the same.")
        # change type to be np.ndarray if it's not already
        self.rotor_positions = self._list_to_ndarray(*rotor_positions)
        self.rotor_rotations, = self._list_to_ndarray(rotor_rotations)
        self.rotor_array_set = True
        return None

    def set_wake(self, wake_speed: float, wake_length: float, resolution: int) -> None:
        """
        Sets the three parameters of the wake of this instance.
        :param wake_speed: the speed at which the wake is convected in m/s
        :param wake_length: the downwind length of the wake in m
        :param resolution: the number of points used to discretise one trailing vortex
        :return:
        """
        self._set(**{param: value for param, value in locals().items() if param != "self"})

        # 'wake_set' is internally used to provide meaningful error messages if the user wants to perform
        # calculations that require the rotor properties to be set.
        self.wake_set = True
        return None

    def set_control_points_on_quarter_chord(self) -> None:
        """
        Automatically sets the control points on the bound vortex in the radial middle of a blade element.
        ATTENTION: This is generally NOT the same as the quarter chord of the radial middle of the blade element. It
        is only the same if the blade rotation is constant along the blade. Otherwise, the linear connection of the
        bound vortices between the blade element ends causes the middle of the bound vortices to NOT lie on the
        actual quarter chord point of the middle of the blade element. This is because the actual quarter chord point
        follows the proper angles while the bound vortex connection is a straight line.
        ATTENTION 2: If a rotor array is used, this function must be called after 'set_rotor_array' if the first
        rotor has a rotation.
        :return:
        """
        rotor_angle = 0 if self.rotor_rotations is None else self.rotor_rotations[0]
        qc_elements = self.c_elements/4 # quarter chord of blade element ends
        x_qc = -np.sin(self.blade_rotation)*qc_elements # x position of the quarter chord of blade element ends
        y_qc = np.cos(self.blade_rotation)*qc_elements-np.sin(rotor_angle)*self.r_elements # y position of the
        # quarter chord of blade element ends
        z_qc = np.cos(rotor_angle)*self.r_elements # z position of the quarter chord of the blade ends
        coordinates_ends = np.asarray([[x, y, z] for x, y, z in zip(x_qc, y_qc, z_qc)])
        # the line below assumes the control point to be in the middle (x, y, and z wise) between the two ends,
        # i.e. on the bound vortex in the radial middle of the element.
        self.control_points = (coordinates_ends[:-1]+coordinates_ends[1:])/2
        self.n_control_points = self.control_points.shape[0]
        return None

    def set_control_points(self,
                           x_control_points: float or np.ndarray,
                           y_control_points: float or np.ndarray,
                           z_control_points: float or np.ndarray) -> None:
        """
        Sets the control point coordinates for the blade.

        The control points are specified by their individual coordinates. If an array of coordinates is given then
        that array must be a single row array. The control points are internally saved as a list of
        (1, 3) sized np.ndarrays.
        :param x_control_points: float or np.ndarray of x coordinates
        :param y_control_points: -"- y coordinates
        :param z_control_points: -"- z coordinates
        :return: None
        """
        # check how many coordinates are given per axis
        n_x = 1 if type(x_control_points) == float or type(x_control_points) == int else x_control_points.size
        n_y = 1 if type(y_control_points) == float or type(y_control_points) == int else y_control_points.size
        n_z = 1 if type(z_control_points) == float or type(z_control_points) == int else z_control_points.size
        lengths = np.asarray([n_x, n_y, n_z])
        if np.unique(lengths).size > 2 or (np.unique(lengths).size > 1 and 1 not in lengths): # check if dimensions
            # match
            raise ValueError(f"Number of coordinates for the control points don't match. Input lengths are [n_x n_y "
                             f"n_z] = {lengths}")

        self.n_control_points = np.max([n_x, n_y, n_z]) # get number of control points
        x_cp, y_cp, z_cp = self._float_to_ndarray(self.n_control_points, x_control_points, y_control_points,
                                                  z_control_points)
        self.control_points = [np.asarray([x, y, z]) for x, y, z in zip(x_cp, y_cp, z_cp)] # update structure for later
        # use
        return None

    def blade(self) -> None:
        """
        Calculates the vortex system (as in coordinates of the line vortices) of a blade, meaning its trailing and
        bound vortex system.
        :return: None
        """
        self._assert_properties("blade", self.blade) # assert that the blade properties have been set
        self._blade_bound()                          # calculate the bound vortex coordinates for the blade
        self._assert_properties("wake", self.blade)  # assert that the wake properties have been set
        self._blade_trailing()                       # calculate the trailing vortex coordinates for the blade
        self.blade_calculated = True                 # for later error clarification
        return None

    def rotor(self) -> None:
        """
        Calculates the vortex system (as in coordinates of the line vortices) of the rotor, meaning its trailing and
        bound vortex system.
        :return:
        """
        self._assert_properties("blade", self.rotor) # assert that the blade properties have been set
        self._rotor_bound()                          # calculate the bound vortex coordinates for the rotor
        self._assert_properties("wake", self.rotor)  # assert that the wake properties have been set
        self._rotor_trailing()                       # calculate the trailing vortex coordinates for the rotor
        self.rotor_calculated = True                 # for later error clarification
        return None

    def rotor_array(self) -> None:
        self._assert_properties("array", self.rotor_array) # assert that the rotor array properties are defined
        self._assert_properties("blade", self.rotor_array) # assert that the blade properties are defined
        self._rotor_array_bound(self.rotor_positions, self.rotor_rotations) # calculate the bound vortex coordinates
        # of the rotor array
        self._assert_properties("wake", self.rotor_array) # assert that the wake properties are defined
        self._rotor_array_trailing(self.rotor_positions, self.rotor_rotations) # calculate the trailing vortex
        # coordinates of the rotor array
        self.rotor_array_calculated = True # for later error clarification
        return None

    def bound_induction_matrices(self,
                                 vortex_core_radius: float,
                                 vortex_system_type: str="rotor_array") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the induction matrix from bound vortices of the 'vortex_sytem_type' on all control points. The
        columns of the induction matrices correspond to a certain circulation, while the rows correspond to a certain
        control point. In general: the entry (i,j) (row, column) of an induction matrix calculates (with a given
        circulation vector) the induced velocity of circulation 'j' on control point 'i'.
        :return: tuple with the induction matrices which are numpy arrays
        :param vortex_core_radius: Radius of the vortex core
        :param vortex_system_type: Which vortex system to use. Valid choices are: "blade", "rotor", and "rotor_array"
        :return: induction matrices (numpy arrays) in the three axes directions given as a tuple.
        """
        if vortex_system_type not in ["rotor_array", "rotor", "blade"]:
            raise ValueError(f"Supported vortex system types are ['rotor_array', 'rotor', 'blade']. You tried it with "
                             f"{vortex_system_type}.")
        self._assert_vortex_system(vortex_system_type)  # assert that the needed vortex coordinates have been calculated

        # The later calculation of the induction factors always assumes a calculated rotor array. However,
        # if no rotor array has been calculated then a rotor array is artificially created. If, e.g.,
        # only the vortex system of a blade was calculated and shall be examined, then the following lines create a
        # rotor array that consists only of that blade's vortex system.
        coordinates_rotor_array_bound_rotorwise = self.coordinates_rotor_array_bound_rotorwise
        if vortex_system_type == "rotor":
            coordinates_rotor_array_bound_rotorwise = self.coordinates_rotor_bound_bladewise  # only use one rotor
        elif vortex_system_type == "blade":
            coordinates_rotor_array_bound_rotorwise = [self.coordinates_blade_bound]  # only use the blade

        n_circulations = len(self.r_elements)-1  # number of bound circulations
        n_rotors = len(coordinates_rotor_array_bound_rotorwise)  # number of rotors in the rotor array
        single_trailing_induction_matrices = {  # the inductions are calculated for individual bound vortex systems
            # first (debugging help)
            "x": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)],
            "y": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)],
            "z": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)]
        }
        for i_rotor, rotor_bound in enumerate(coordinates_rotor_array_bound_rotorwise):  # every rotor
            for i_blade in range(self.n_blades):  # every blade of the current rotor
                i_bound_system = i_rotor*self.n_blades+i_blade  # number of the current bound vortex system
                blade_bound = rotor_bound[i_blade*n_circulations:(i_blade+1)*n_circulations, :]  # coordinates of the
                # current bound vortex system
                for cp_i, control_point in enumerate(self.control_points):  # every control point
                    for vortex_i, (vortex_start, vortex_end) in enumerate(zip(blade_bound[:-1], blade_bound[1:])):
                        # go over each vortex filament of the bound vortices.
                        induction_factors = self._vortex_induction_factor(vortex_start,
                                                                          vortex_end,
                                                                          control_point,
                                                                          vortex_core_radius)
                        single_trailing_induction_matrices["x"][i_bound_system][cp_i, vortex_i] = induction_factors[0]
                        single_trailing_induction_matrices["y"][i_bound_system][cp_i, vortex_i] = induction_factors[1]
                        single_trailing_induction_matrices["z"][i_bound_system][cp_i, vortex_i] = induction_factors[2]

        induction_matrices = {  # initialise container for the final, summed up induction factor matrices
            "x": np.zeros((self.n_control_points, n_circulations)),
            "y": np.zeros((self.n_control_points, n_circulations)),
            "z": np.zeros((self.n_control_points, n_circulations))
        }
        for direction in ["x", "y", "z"]:
            for induction_mat in single_trailing_induction_matrices[direction]:
                induction_matrices[direction] += induction_mat  # sum the contributions of all blade wakes
        return induction_matrices["x"], induction_matrices["y"], induction_matrices["z"]

    def trailing_induction_matrices(self,
                                    vortex_core_radius: float,
                                    vortex_system_type: str="rotor_array") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the induction matrix from trailing vortices of the 'vortex_sytem_type' on all control points. The
        columns of the induction matrices correspond to a certain circulation, while the rows correspond to a certain
        control point. In general: the entry (i,j) (row, column) of an induction matrix calculates (with a given
        circulation vector) the induced velocity of circulation 'j' on control point 'i'.
        :return: tuple with the induction matrices which are numpy arrays
        :param vortex_core_radius: Radius of the vortex core
        :param vortex_system_type: Which vortex system to use. Valid choices are: "blade", "rotor", and "rotor_array"
        :return: induction matrices (numpy arrays) in the three axes directions given as a tuple.
        """
        if vortex_system_type not in ["rotor_array", "rotor", "blade"]:
            raise ValueError(f"Supported vortex system types are ['rotor_array', 'rotor', 'blade']. You tried it with "
                             f"{vortex_system_type}.")
        self._assert_vortex_system(vortex_system_type)  # assert that the needed vortex coordinates have been calculated

        # The later calculation of the induction factors always assumes a calculated rotor array. However,
        # if no rotor array has been calculated then a rotor array is artificially created. If, e.g.,
        # only the vortex system of a blade was calculated and shall be examined, then the following lines create a
        # rotor array that consists only of that blade's vortex system.
        coordinates_rotor_array_trailing_rotorwise = self.coordinates_rotor_array_trailing_rotorwise
        if vortex_system_type == "rotor":
            coordinates_rotor_array_trailing_rotorwise = self.coordinates_rotor_trailing_bladewise
        elif vortex_system_type == "blade":
            coordinates_rotor_array_trailing_rotorwise = [self.coordinates_blade_trailing]

        n_circulations = len(self.r_elements) # because at each blade element a vortex line is shed (which has a
        # constant circulation!)
        n_rotors = len(coordinates_rotor_array_trailing_rotorwise) # number of rotors in the array
        single_trailing_induction_matrices = { # the inductions are calculated for individual trailing vortex systems
            # first (debugging help)
            "x": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)],
            "y": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)],
            "z": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades*n_rotors)]
        }

        for i_rotor, rotor_trailing in enumerate(coordinates_rotor_array_trailing_rotorwise):  # every rotor
            for i_blade in range(self.n_blades):  # every blade
                i_wake_system = i_rotor*self.n_blades+i_blade  # number of the current trailing vortex system
                n_coords_per_blade = n_circulations*(self.resolution+1)  # number of the trailing vortex coordinates
                # per blade
                blade_trailing = rotor_trailing[i_blade*n_coords_per_blade:(i_blade+1)*n_coords_per_blade, :]
                # all trailing vortex coordinates that belong to the current blade
                for cp_i, control_point in enumerate(self.control_points): # iterate over the control points
                    for inducing_element in range(n_circulations): # iterate over the trailing vortex of each blade
                        # element. Every blade element thus influences (thus inducing_element) the current element

                        # get the start points of each vortex element of the trailing vortex from the inducing blade
                        # element
                        vortex_starts = blade_trailing[inducing_element*(self.resolution+1):
                                                       (inducing_element+1)*(self.resolution+1)-1]
                        # get the end points of each vortex element of the trailing vortex from the inducing blade
                        # element
                        vortex_ends = blade_trailing[1+inducing_element*(self.resolution+1):
                                                     (inducing_element+1)*(self.resolution+1)]
                        induction_factors = np.zeros(3)
                        for vortex_start, vortex_end in zip(vortex_starts, vortex_ends):  # iterate over all vortex
                            # elements
                            induction_factors += self._vortex_induction_factor(vortex_start,
                                                                               vortex_end,
                                                                               control_point,
                                                                               vortex_core_radius)
                        # place the induction factors (x, y, and z component) of this wake in its respective wake
                        # induction matrix
                        single_trailing_induction_matrices["x"][i_wake_system][cp_i, inducing_element] = induction_factors[0]
                        single_trailing_induction_matrices["y"][i_wake_system][cp_i, inducing_element] = induction_factors[1]
                        single_trailing_induction_matrices["z"][i_wake_system][cp_i, inducing_element] = induction_factors[2]

        induction_matrices = { # initialise container for the final, summed together induction factor matrices
            "x": np.zeros((self.n_control_points, n_circulations)),
            "y": np.zeros((self.n_control_points, n_circulations)),
            "z": np.zeros((self.n_control_points, n_circulations))
        }
        for direction in ["x", "y", "z"]:
            for induction_mat in single_trailing_induction_matrices[direction]:
                induction_matrices[direction] += induction_mat # sum the contributions of all blade wakes
        return induction_matrices["x"], induction_matrices["y"], induction_matrices["z"]

    def blade_elementwise_visualisation(self,
                                        trailing: bool=True,
                                        bound: bool=True,
                                        control_points: bool=False) -> None:
        """
        Visualises the wake of one blade. Colours the trailing vortex of each blade element separately. If control
        points have been set already, those can be visualised as well.
        :param trailing: Visualise the trailing vortices
        :param bound: Visualise the bound vortices
        :param control_points: Visualise the control points
        :return: None
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        if trailing:
            for r in self.r_elements:
                ax.plot(self.coordinates_blade_trailing_elementwise["x"][r],
                        self.coordinates_blade_trailing_elementwise["y"][r],
                        self.coordinates_blade_trailing_elementwise["z"][r])
        if bound:
            x_bound = self.coordinates_blade_bound_elementwise["x"]["_"]
            y_bound = self.coordinates_blade_bound_elementwise["y"]["_"]
            z_bound = self.coordinates_blade_bound_elementwise["z"]["_"]
            x_start, x_end = x_bound[:-1], x_bound[1:]
            y_start, y_end = y_bound[:-1], y_bound[1:]
            z_start, z_end = z_bound[:-1], z_bound[1:]
            for x_s, x_e, y_s, y_e, z_s, z_e in zip(x_start, x_end, y_start, y_end, z_start, z_end):
                ax.plot((x_s, x_e), (y_s, y_e), (z_s, z_e))

        if control_points:
            if self.control_points is None:
                raise ValueError("No control points can be visualised because none have been set. Set them using "
                                 "'set_control_points()' first.")
            for control_point in self.control_points:
                ax.plot(control_point[0], control_point[1], control_point[2], "ko")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
        return None

    def rotor_visualisation(self,
                            trailing: bool=True,
                            bound: bool=True,
                            control_points: bool=False,
                            show: bool=True) -> None or tuple[plt.figure, plt.axes]:
        """
        Visualises the wake of the whole rotor. Currently supports a maximum of 7 wakes (due to colouring).
        :param trailing: Visualise the trailing vortices
        :param bound: Visualise the bound vortices
        :param control_points: Visualise the control points
        :param show: whether to show the plot. If False, the pyplot figure and axis are returned.
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        colours = ["b", "g", "r", "c", "m", "y", "k"]
        n_elements = len(self.r_elements)
        if trailing:
            for trailing, c in zip(self.coordinates_rotor_trailing_bladewise, colours):
                for element in range(n_elements):
                    ax.plot(trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 0],
                            trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 1],
                            trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 2], color=c)
        if bound:
            for bound, c in zip(self.coordinates_rotor_bound_bladewise, colours):
                for element in range(n_elements):
                    ax.plot(bound[element*n_elements:(element+1)*n_elements, 0],
                            bound[element*n_elements:(element+1)*n_elements, 1],
                            bound[element*n_elements:(element+1)*n_elements, 2], color=c)

        if control_points:
            if self.control_points is None:
                raise ValueError("No control points can be visualised because none have been set. Set them using "
                                 "'set_control_points()' first.")
            for control_point in self.control_points:
                ax.plot(control_point[0], control_point[1], control_point[2], "ko")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if show:
            plt.show()
            return None
        else:
            return fig, ax

    def rotor_array_visualisation(self,
                                  trailing: bool=True,
                                  bound: bool=True,
                                  control_points: bool=False,
                                  show: bool=True) -> None or tuple[plt.figure, plt.axes]:
        """
        Visualises the wake of the whole rotor array. Currently supports a maximum of 7 rotors (due to colouring).
        :param trailing: Visualise the trailing vortices
        :param bound: Visualise the bound vortices
        :param control_points: Visualise the control points
        :param show: whether to show the plot. If False, the pyplot figure and axis are returned.
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        colours = ["b", "g", "r", "c", "m", "y", "k"]
        n_blade_elements = len(self.r_elements)
        if trailing:
            for coordinates_rotor, c in zip(self.coordinates_rotor_array_trailing_rotorwise, colours):
                for i_blade in range(self.n_blades):
                    n_coords_per_blade = n_blade_elements*(self.resolution+1)
                    coordinates = coordinates_rotor[i_blade*n_coords_per_blade:(i_blade+1)*n_coords_per_blade, :]
                    for element in range(n_blade_elements):
                        ax.plot(coordinates[element*(self.resolution+1):(element+1)*(self.resolution+1), 0],
                                coordinates[element*(self.resolution+1):(element+1)*(self.resolution+1), 1],
                                coordinates[element*(self.resolution+1):(element+1)*(self.resolution+1), 2], color=c)

        if bound:
            for coordinates_rotor, c in zip(self.coordinates_rotor_array_bound_rotorwise, colours):
                for element in range(n_blade_elements):
                    ax.plot(coordinates_rotor[element*n_blade_elements:(element+1)*n_blade_elements, 0],
                            coordinates_rotor[element*n_blade_elements:(element+1)*n_blade_elements, 1],
                            coordinates_rotor[element*n_blade_elements:(element+1)*n_blade_elements, 2], color=c)

        if control_points:
            if self.control_points is None:
                raise ValueError("No control points can be visualised because none have been set. Set them using "
                                 "'set_control_points()' first.")
            for control_point in self.control_points:
                ax.plot(control_point[0], control_point[1], control_point[2], "ko")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if show:
            plt.show()
            return None
        else:
            return fig, ax

    def _blade_bound_elementwise(self) -> dict[str, dict]:
        """
        Creates a dictionary of the bound vortex coordinates. Each key is an axis and its values are np.ndarrays with
        the coordinates.
        :return: None
        """
        qc_elements = self.c_elements/4 # quarter chord
        x_qc = -np.sin(self.blade_rotation)*qc_elements # x position of the quarter chord
        y_qc = np.cos(self.blade_rotation)*qc_elements # y position of the quarter chord
        self.coordinates_blade_bound_elementwise = {"x": {"_": x_qc}, "y": {"_": y_qc}, "z": {"_": self.r_elements}}
        # the second layer dictionaries have to be artificially used to work with '_combine_elements'.
        return self.coordinates_blade_bound_elementwise

    def _blade_trailing_elementwise(self) -> dict[str, dict]:
        """
        Creates a dictionary of the trailing vortices coordinates. Each key (representing an axis) of the final
        dictionary has a dictionary as its value. These second layer dictionaries have each blade element's radius as
        key and (x, y, or y depending on the first layer dictionary) the trailing vortices' coordinates that belong
        to that blade element.
        :return: None
        """
        qc_elements = self.c_elements/4 # quarter chord
        x_qc = -np.sin(self.blade_rotation)*qc_elements # x position of the quarter chord
        y_qc = np.cos(self.blade_rotation)*qc_elements # y position of the quarter chord
        x_c = -np.sin(self.blade_rotation)*self.c_elements # x position of the trailing edge
        y_c = np.cos(self.blade_rotation)*self.c_elements # y position of the trailing edge
        # the swept vortices start a quarter chord behind the airfoil
        x_swept_trailing_vortices_start = x_c+x_qc # x position of the first node of the first swept trailing vortex
        y_swept_trailing_vortices_start = y_c+y_qc # y position of the first node of the first swept trailing vortex
        # the following three lines define the containers for the trailing vortex structure of each blade element
        # they are initialised with the fixed trailing vortex (first trailing vortex = fixed trailing vortex)
        x = {r: [x_qc[i], x_swept_trailing_vortices_start[i]] for i, r in enumerate(self.r_elements)}
        y = {r: [y_qc[i], y_swept_trailing_vortices_start[i]] for i, r in enumerate(self.r_elements)}
        z = {r: [r, r] for r in self.r_elements}
        for t in np.linspace(self.wake_length/(self.wake_speed*self.resolution),
                             self.wake_length/self.wake_speed,
                             self.resolution-1):
            angle = -self.rotor_rotation_speed*t+np.pi/2 # It is assumed that the rotor is rotating clockwise when
            # looking downwind. The +np.pi/2 rotate the t=0 position of the blade to be parallel to the z-axis.
            for i, r in enumerate(self.r_elements):
                x[r].append(x_swept_trailing_vortices_start[i]+self.wake_speed*t) # pure convection
                # the first trailing vortex is parallel to the chord
                y[r].append(np.sin(angle)*y_swept_trailing_vortices_start[i]+r*np.cos(angle)) # rotated
                z[r].append(-np.cos(angle)*y_swept_trailing_vortices_start[i]+r*np.sin(angle)) # rotated
        self.coordinates_blade_trailing_elementwise = {"x": x, "y": y, "z": z}
        return self.coordinates_blade_trailing_elementwise

    def _combine_elementwise_blade(self, coordinates_from, if_not_do, combine_to: str) -> None:
        """
        Combines element-wise coordinates into a np.ndarray of size (N,3).
        :param coordinates_from:    The element-wise coordinates of the vortex system that are to be combined
        :param if_not_do:           Function that calculates these elementwise coordinates if they are not yet
                                    calculated
        :param save_to:             Class property name as string (without self.) to which the combined coordinates are
                                    saved.
        :return: None
        """
        if coordinates_from is None: # if the element-wise coordinates do not yet exist
            coordinates_from = if_not_do() # then calculate them
        self.__dict__[combine_to] = np.asarray([ # save combined coordinates to a np.ndarray of size (N,3)
            [coord for coords in coordinates_from["x"].values() for coord in coords], # this appends all values from
            [coord for coords in coordinates_from["y"].values() for coord in coords], # the second layer dictionaries
            [coord for coords in coordinates_from["z"].values() for coord in coords], # to one another (axis-wise).
        ]).T
        return None

    def _rotate_combined_blade(self,
                               coordinates_from,
                               if_not_do,
                               rotate_to: str) -> None:
        """
        Creates a specific (bound or trailing) vortex system for the specified rotor.
        It does that by taking the specific vortex system from a single blade and rotating it 'n_blades'-1 times. The
        resulting vortex system is saved as a list with the vortex system of each blade as a unique entry. Each entry is
        a numpy array of size (N,3). Each row corresponds to one vortex line and the columns correspond to the three
        axes. N: roughly blade elements times resolution of the trailing vortices.

        :param coordinates_from:    The coordinates of the vortex system that is to be rotated
        :param if_not_do:           Function that calculates the combined coordinates of the vortex system if they are
                                    not yet calculated
        :param rotate_to:           Class property name as string (without self.) to which the combined coordinates are
                                    saved.
        :return:
        """
        if coordinates_from is None: # if the combined coordinates do not yet exist
            coordinates_from = if_not_do() # then calculate them
        self.__dict__[rotate_to] = [coordinates_from] # start list with the un-rotated blade vortex system
        for rot_angle in np.linspace(2*np.pi/self.n_blades, 2*np.pi*(1-1/self.n_blades), self.n_blades-1): #
            # calculate the angular positions of the other blades
            rot_matrix = np.array([[1, 0, 0], # calculate rotation matrix for the current blade
                                   [0, np.cos(rot_angle), np.sin(rot_angle)],
                                   [0, -np.sin(rot_angle), np.cos(rot_angle)]])
            self.__dict__[rotate_to].append(np.dot(coordinates_from, rot_matrix)) # add rotated vortex system
        return None

    def _combine_rotated_blades(self, coordinates_from, if_not_do, combine_to: str) -> None:
        """
        Combines blade-wise coordinates into a np.ndarray of size (N,3).
        :param coordinates_from:    The blade-wise coordinates of the vortex system that are to be combined
        :param if_not_do:           Function that calculates these blade-wise coordinates if they are not yet
                                    calculated
        :param save_to:             Class property name as string (without self.) to which the combined coordinates are
                                    saved.
        :return: None
        """
        if coordinates_from is None: # if the element-wise coordinates do not yet exist
            coordinates_from = if_not_do() # then calculate them
        self.__dict__[combine_to] = np.asarray([c.tolist() for c_bladewise in coordinates_from for c in c_bladewise])
        return None

    def _place_rotors(self, coordinates, combine_to: str, rotor_positions: list[np.ndarray],
                      rotor_rotations: np.ndarray) -> None:
        self.__dict__[combine_to] = list()
        for position, rotation in zip(rotor_positions, rotor_rotations):
            rot_matrix = np.array([[1, 0, 0], # calculate rotation matrix for the current blade
                                   [0, np.cos(rotation), np.sin(rotation)],
                                   [0, -np.sin(rotation), np.cos(rotation)]])
            self.__dict__[combine_to].append(np.dot(coordinates, rot_matrix)+position)  # and shift to proper position
        return None

    def _blade_bound(self) -> np.ndarray:
        """
        Combines the element-wise bound vortex system for one blade as one data structure. The logic is explained in
        _combine_elementwise_blade.
        :return: points of the bound vortices of a blade
        """
        self._combine_elementwise_blade(coordinates_from=self.coordinates_blade_bound_elementwise,
                                        if_not_do=self._blade_bound_elementwise,
                                        combine_to="coordinates_blade_bound")
        return self.coordinates_blade_bound

    def _blade_trailing(self) -> np.ndarray:
        """
        Combines the element-wise trailing vortex system for one blade into one np.ndarray. The logic is explained in
        _combine_elementwise_blade.
        :return: points of the trailing vortices of a blade
        """
        self._combine_elementwise_blade(coordinates_from=self.coordinates_blade_trailing_elementwise,
                                        if_not_do=self._blade_trailing_elementwise,
                                        combine_to="coordinates_blade_trailing")
        return self.coordinates_blade_trailing

    def _rotor_bound(self) -> np.ndarray:
        """
        Creates the full bound vortex system of a rotor. The logic is explained in _rotate_combined_blade.
        :return: None
        """
        self._rotate_combined_blade(coordinates_from=self.coordinates_blade_bound,
                                    if_not_do=self._blade_bound,
                                    rotate_to="coordinates_rotor_bound_bladewise")
        return self.coordinates_rotor_bound_bladewise

    def _rotor_trailing(self) -> np.ndarray:
        """
        Creates the full trailing vortex system of a rotor. The logic is explained in _rotate_combined_blade.
        :return: None
        """
        self._rotate_combined_blade(coordinates_from=self.coordinates_blade_trailing,
                                    if_not_do=self._blade_trailing,
                                    rotate_to="coordinates_rotor_trailing_bladewise")
        return self.coordinates_rotor_trailing_bladewise

    def _rotor_array_bound(self,
                           rotor_positions: list[np.ndarray],
                           rotor_rotations: np.ndarray) -> None:
        self._combine_rotated_blades(coordinates_from=self.coordinates_rotor_bound_bladewise,
                                     if_not_do=self._rotor_bound,
                                     combine_to="coordinates_rotor_bound")
        self._place_rotors(self.coordinates_rotor_bound, combine_to="coordinates_rotor_array_bound_rotorwise",
                           rotor_positions=rotor_positions, rotor_rotations=rotor_rotations)
        return None

    def _rotor_array_trailing(self,
                              rotor_positions: list[np.ndarray],
                              rotor_rotations: np.ndarray) -> None:
        self._combine_rotated_blades(coordinates_from=self.coordinates_rotor_trailing_bladewise,
                                     if_not_do=self._rotor_trailing,
                                     combine_to="coordinates_rotor_trailing")
        self._place_rotors(self.coordinates_rotor_trailing, combine_to="coordinates_rotor_array_trailing_rotorwise",
                           rotor_positions=rotor_positions, rotor_rotations=rotor_rotations)
        return None

    @staticmethod
    def _vortex_induction_factor(vortex_start: np.ndarray,
                                 vortex_end: np.ndarray,
                                 induction_point: np.ndarray,
                                 core_radius: float) -> np.ndarray:
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
        r_s = induction_point-vortex_start # vector from the start of the vortex to the induction point
        r_e = induction_point-vortex_end # vector from the end of the vortex to the induction point
        r_v = vortex_end-vortex_start # vector from the start of the vortex to the end of the vortex

        l_s = np.linalg.norm(r_s) # distance between the induction point and the start of the vortex
        l_e = np.linalg.norm(r_e) # distance between the induction point and the end of the vortex
        l_v = np.linalg.norm(r_v) # length of the vortex

        h = np.linalg.norm(np.cross(r_v, r_s))/l_v # shortest distance between the control point and an infinite
        # extension of the vortex filament
        if h == 0: # the control point lies in the centre of the vortex
            return np.zeros(3)

        e_i = np.cross(r_v, r_s)/(h*l_v) # unit vector of the direction of induced velocity
        if h <= core_radius: # the control point lies inside the vortex core
            return h/(2*np.pi*core_radius**2)*e_i # induced velocity of solid body rotation
        else:
            return e_i/(4*np.pi*h*l_v)*(np.dot(r_v, (r_s/l_s-r_e/l_e))) # induced velocity of irrotational vortex

    def _set(self, **kwargs) -> None:
        """
        Sets any parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
        :param kwargs:
        :return:
        """
        existing_parameters = [*self.__dict__]
        for parameter, value in kwargs.items(): # puts the tuples of parameters and values
            if parameter not in existing_parameters:
                raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
            self.__dict__[parameter] = value
        return None

    def _assert_properties(self, check: str, called_from) -> None:
        to_check = {
            "blade": [self.rotor_set, "set_blade()"],
            "wake": [self.wake_set, "set_wake()"],
            "array": [self.rotor_array_set, "set_rotor_array()"]
        }
        if not to_check[check][0]:
            raise ValueError(f"{check} properties have to be set using {to_check[check][1]} before calling "
                             f"{called_from.__name__}().")
        return None

    def _assert_vortex_system(self, vortex_system_type: str):
        response = {
            "blade": [self.blade_calculated, "blade(), rotor(), or rotor_array()"],
            "rotor": [self.rotor_calculated, "rotor(), or rotor_array()"],
            "rotor_array": [self.rotor_array_calculated, "rotor_array()"]
        }
        if response[vortex_system_type][0] is None:
            raise ValueError(f"A {vortex_system_type} trailing vortex system has to be calculated first. Use"
                             f" {response[vortex_system_type][1]} first.")
        return None

    @staticmethod
    def _float_to_ndarray(length: int, *args) -> list[np.ndarray]:
        """
        Loops through the input arguments to ensure they are numpy arrays. If a float is given, it is turned into an
        array of length 'length' with all values being the initial float.
        """
        return [arg if type(arg) == np.ndarray else np.asarray([arg for _ in range(length)]) for arg in args]

    @staticmethod
    def _list_to_ndarray(*args) -> list[np.ndarray]:
        """
        Checks if 'args' are lists or numpy ndarrays. If they are a list, convert them to numpy ndarrays. Returns a
        list of all converted 'args'
        :param args: lists or numpy ndarrays
        :return: list of numpy ndarrays
        """
        return  [arg if type(arg) == np.ndarray else np.asarray(arg) for arg in args]

