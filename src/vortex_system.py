# class to create the vortex system of a wing, a blade or a whole rotor
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


class VortexSystem:
    def __init__(self):
        """
        Initialize empty object
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

        # control points
        self.control_points = None
        self.n_control_points = None

        # instance properties
        # bound vortices
        self.coordinates_blade_bound_elementwise = None
        self.coordinates_blade_bound = None
        self.coordinates_rotor_bound = None
        # trailing vortices
        self.coordinates_blade_trailing_elementwise = None
        self.coordinates_blade_trailing = None
        self.coordinates_rotor_trailing = None

        # error clarification
        self.rotor_set = False
        self.wake_set = False

    def set_blade(self,
                  r_elements: np.ndarray,
                  c_elements: float or np.ndarray,
                  blade_rotation: float or np.ndarray,
                  rotor_rotation_speed: float,
                  n_blades: int = 3
                  ) -> None:
        """
        Set the blade geometry parameters (blade discretization, chord per element, twist/pitch, number of blades), and the rotation speed 

        The rotor coordinate system is such that the x faces downwind along the rotor axis, y faces to the left
        facing the rotor from the front (i.e. looking downwind) and z point upwards. z goes along the leading edge of a
        blade that is point upward. All angles follow a standard right-hand rule.

        :param r_elements:              blade section radial position
        :param c_elements:              blade section chord length 
        :param blade_rotation:          rotation of the blade sections around the z-axis in radians. An angle of 0
        means the chord is parallel to the y-axis with the trailing edge pointing in the positive y-direction.
        :param rotor_rotation_speed:    Rotational speed of the rotor
        :param n_blades:                Number of blades
        :return:                        None
        """
        self.r_elements = r_elements
        # checks if a list of chord and blade rotations is given. if not, create a list with the uniform value using the _float_to_ndarray function
        self.c_elements, self.blade_rotation = self._float_to_ndarray(len(r_elements), c_elements, blade_rotation)
        # Ensure that the provided arrays have the same length as the elment array, raise error otherwise
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
        self.coordinates_rotor_bound = None
        # trailing vortex system
        self.coordinates_blade_trailing_elementwise = None
        self.coordinates_blade_trailing = None
        self.coordinates_rotor_trailing = None
        # error clarification for user
        self.rotor_set = True
        return None

    def set_wake(self, wake_speed: float, wake_length: float, resolution: int) -> None:
        """
        Calls a function to define the wake and change the wake status.
        
        What is the dimensions here? is the wake length in diameter or in length units?
        What resolution is this? Along the blade span or along a trailing vortex?
        """
        self._set(**{param: value for param, value in locals().items() if param != "self"})
        # Set a property of the object to indicate, whether the wake is already defined 
        # error clarification for user
        self.wake_set = True
        return None

    def set_control_points_on_quarter_chord(self) -> None:
        """
        Automatically sets the control points on the quarter chord and the radial middle of the blade elements.
        :return:
        """
        qc_elements = self.c_elements/4 # quarter chord of blade element ends
        x_qc = -np.sin(self.blade_rotation)*qc_elements # x position of the quarter chord of blade element ends
        y_qc = np.cos(self.blade_rotation)*qc_elements # y position of the quarter chord of blade element ends
        coordinates_ends = np.asarray([[x, y, z] for x, y, z in zip(x_qc, y_qc, self.r_elements)])
        # the line below assumes the control point to be in the middle (x, y, and z wise) between the two ends.
        self.control_points = (coordinates_ends[:-1]+coordinates_ends[1:])/2
        self.n_control_points = self.control_points.shape[0]
        return None

    def set_control_points(self,
                           x_control_points: float or np.ndarray,
                           y_control_points: float or np.ndarray,
                           z_control_points: float or np.ndarray) -> None:
        """
        Sets the control point coordinates for the blade 

        The control points are specified by their individual coordinates. If an array of coordinates is given then
        that array must be a single row or single column array. The control points are internally saved as a list of
        (1,3) sized np.ndarrays.
        :param x_control_points: list/array of x coordinates 
        :param y_control_points: -"- y coordinates
        :param z_control_points: -"- z coordinates
        :return: None
        """
        # check how many coordinates are given per axis
        n_x = 1 if type(x_control_points) == float or type(x_control_points) == int else x_control_points.size
        n_y = 1 if type(y_control_points) == float or type(y_control_points) == int else y_control_points.size
        n_z = 1 if type(z_control_points) == float or type(z_control_points) == int else z_control_points.size
        lengths = np.asarray([n_x, n_y, n_z])
        if np.unique(lengths).size > 2 or (np.unique(lengths).size > 1 and 1 not in lengths): # check if dimensions match
            raise ValueError(f"Number of coordinates for the control points don't match. Input lengths are [n_x n_y "
                             f"n_z] = {lengths}")

        self.n_control_points = np.max([n_x, n_y, n_z]) # get number of control points
        x_cp, y_cp, z_cp = self._float_to_ndarray(self.n_control_points, x_control_points, y_control_points,
                                                  z_control_points)
        self.control_points = [np.asarray([x, y, z]) for x, y, z in zip(x_cp, y_cp, z_cp)] # update structure for later
        # use
        return None

    def blade_trailing(self) -> np.ndarray:
        """
        Combines the element-wise trailing vortex system for one blade into one np.ndarray. The logic is explained in
        _combine_elementwise.
        :return: points of the trailing vortices of a blade
        """
        self._combine_elementwise(called_from=self.blade_trailing,
                                  coordinates_from=self.coordinates_blade_trailing_elementwise,
                                  if_not_do=self._blade_trailing_elementwise,
                                  combine_to="coordinates_blade_trailing")
        return self.coordinates_blade_trailing

    def blade_bound(self) -> np.ndarray:
        """
        Combines the element-wise bound vortex system for one blade as one data structure. The logic is explained in
        _combine_elementwise.
        :return: points of the bound vortices of a blade
        """
        self._combine_elementwise(called_from=self.blade_bound,
                                  coordinates_from=self.coordinates_blade_bound_elementwise,
                                  if_not_do=self._blade_bound_elementwise,
                                  combine_to="coordinates_blade_bound")
        return self.coordinates_blade_bound

    def blade(self) -> None:
        """
        Calculates the vortex system (as in coordinates of the line vortices) of a blade, meaning its trailing and
        bound vortex system.
        :return: None
        """
        self.blade_trailing()
        self.blade_bound()
        return None

    def rotor_trailing(self) -> None:
        """
        Creates the full trailing vortex system of a rotor. The logic is explained in _rotate_combined.
        :return: None
        """
        self._rotate_combined(called_from=self.rotor_trailing,
                              coordinates_from=self.coordinates_blade_trailing,
                              if_not_do=self.blade_trailing,
                              rotate_to="coordinates_rotor_trailing")
        return None

    def rotor_bound(self) -> None:
        """
        Creates the full bound vortex system of a rotor. The logic is explained in _rotate_combined.
        :return: None
        """
        self._rotate_combined(called_from=self.rotor_bound,
                              coordinates_from=self.coordinates_blade_bound,
                              if_not_do=self.blade_bound,
                              rotate_to="coordinates_rotor_bound")
        return None

    def rotor(self) -> None:
        """
        Calculates the vortex system (as in coordinates of the line vortices) of the rotor, meaning its trailing and
        bound vortex system.
        :return:
        """
        self.rotor_trailing()
        self.rotor_bound()
        return None

    def bound_induction_matrices(self, vortex_core_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the induction matrix from bound vortices of the rotor on all control points.
        :return: tuple with the induction matrices
        """
        self._assert_bound("rotor")

        n_circulations = len(self.r_elements)-1
        single_trailing_induction_matrices = {  # the inductions are calculated for individual bound vortex systems
            # first (debugging help)
            "x": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)],
            "y": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)],
            "z": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)]
        }
        for bound_system_i, bound in enumerate(self.coordinates_rotor_bound):
            for cp_i, control_point in enumerate(self.control_points):
                for vortex_i, (vortex_start, vortex_end) in enumerate(zip(bound[:-1], bound[1:])):
                    induction_factors = self._vortex_induction_factor(vortex_start, vortex_end, control_point, vortex_core_radius)
                    single_trailing_induction_matrices["x"][bound_system_i][cp_i, vortex_i] = induction_factors[0]
                    single_trailing_induction_matrices["y"][bound_system_i][cp_i, vortex_i] = induction_factors[1]
                    single_trailing_induction_matrices["z"][bound_system_i][cp_i, vortex_i] = induction_factors[2]

        induction_matrices = {  # initialise container for the final, summed together induction factor matrices
            "x": np.zeros((self.n_control_points, n_circulations)),
            "y": np.zeros((self.n_control_points, n_circulations)),
            "z": np.zeros((self.n_control_points, n_circulations))
        }
        for direction in ["x", "y", "z"]:
            for induction_mat in single_trailing_induction_matrices[direction]:
                induction_matrices[direction] += induction_mat  # sum the contributions of all blade wakes
        return induction_matrices["x"], induction_matrices["y"], induction_matrices["z"]

    def trailing_induction_matrices(self, vortex_core_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the induction matrix from the wake of the rotor on all control points.
        :return: tuple with the induction matrices for u, v and w
        """
        self._assert_trailing("rotor")  # a rotor wake has to exist

        n_circulations = len(self.r_elements) # because at each blade element a vortex line is shed (which has a
        # constant circulation!)
        single_trailing_induction_matrices = { # the inductions are calculated for individual trailing vortex systems
            # first (debugging help)
            "x": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)],
            "y": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)],
            "z": [np.zeros((self.n_control_points, n_circulations)) for _ in range(self.n_blades)]
        }
        for wake_system_i, wake in enumerate(self.coordinates_rotor_trailing): # iterate over the wakes of each blade
            for cp_i, control_point in enumerate(self.control_points): # iterate over the control points
                for inducing_element in range(len(self.r_elements)): # iterate over the trailing vortex of each blade
                    # element. Every blade element thus influences (thus inducing_element) the current element (which
                    # is therefore called induced_element, because it 'receives' an induced velocity)

                    # get the start points of each vortex element of the trailing vortex from the inducing blade element
                    vortex_starts = wake[inducing_element*(self.resolution+1):(inducing_element+1)*(self.resolution+1)-1]
                    # get the end points of each vortex element of the trailing vortex from the inducing blade element
                    vortex_ends = wake[1+inducing_element*(self.resolution+1):(inducing_element+1)*(self.resolution+1)]
                    induction_factors = np.zeros(3)
                    for vortex_start, vortex_end in zip(vortex_starts, vortex_ends): # iterate over all vortex elements
                        induction_factors += self._vortex_induction_factor(vortex_start, vortex_end, control_point, vortex_core_radius)
                    # place the induction factors (x, y, and z component) of this wake in its respective wake
                    # induction matrix
                    single_trailing_induction_matrices["x"][wake_system_i][cp_i, inducing_element] = induction_factors[0]
                    single_trailing_induction_matrices["y"][wake_system_i][cp_i, inducing_element] = induction_factors[1]
                    single_trailing_induction_matrices["z"][wake_system_i][cp_i, inducing_element] = induction_factors[2]

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
        Visualises the wake of one blade. Colours the trailing vortex of each blade element separately.
        Control points can be given as an input to be visualised as well. Their structure needs to be the same as the
        control points have that are output by 'induction_matrices()', meaning a list of arrays of size 3.
        :return: None
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        if trailing:
            self._assert_trailing("blade")
            for r in self.r_elements:  # or choose individual elements
                ax.plot(self.coordinates_blade_trailing_elementwise["x"][r],
                        self.coordinates_blade_trailing_elementwise["y"][r],
                        self.coordinates_blade_trailing_elementwise["z"][r])
        if bound:
            self._assert_bound("blade")
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
        # ax.view_init(90, 90, -90)
        plt.show()
        return None

    def rotor_visualisation(self,
                            trailing: bool=True,
                            bound: bool=True,
                            control_points: bool=False,
                            show: bool=True) -> None or tuple[plt.figure, plt.axes]:
        """
        Visualises the wake of the whole rotor. Currently supports a maximum of 7 wakes (due to colouring).
        Control points can be given as an input to be visualised as well. Their structure needs to be the same as the
        control points have that are output by 'induction_matrices()', meaning a list of arrays of size 3.
        :return: None
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        colours = ["b", "g", "r", "c", "m", "y", "k"]
        n_elements = len(self.r_elements)
        if trailing:
            self._assert_trailing("rotor")
            for trailing, c in zip(self.coordinates_rotor_trailing, colours[:len(self.coordinates_rotor_trailing)]):
                for element in range(n_elements):
                    ax.plot(trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 0],
                            trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 1],
                            trailing[element*(self.resolution+1):(element+1)*(self.resolution+1), 2], color=c)
        if bound:
            self._assert_bound("rotor")
            for bound, c in zip(self.coordinates_rotor_bound, colours[:len(self.coordinates_rotor_trailing)]):
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

    def _set(self, **kwargs) -> None:
        """
        Sets any parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
        :param kwargs:
        :return:
        """
        existing_parameters = [*self.__dict__]
        for parameter, value in kwargs.items(): # puts the touples of parameters and values
            if parameter not in existing_parameters:
                raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
            self.__dict__[parameter] = value
        return None

    def _assert_properties(self, fnc):
        for params, was_set in {"rotor": self.rotor_set, "wake": self.wake_set}.items():
            if not was_set:
                raise ValueError(f"{params} properties have to be set before using {fnc.__name__}().")
        return None

    def _assert_trailing(self, vortex_system_type):
        response = {
            "blade": [self.coordinates_blade_trailing, "rotor_trailing(), blade_trailing(), rotor(), or blade()"],
            "rotor": [self.coordinates_rotor_trailing, "rotor_trailing() or rotor()"]
        }
        if response[vortex_system_type][0] is None:
            raise ValueError(f"A {vortex_system_type} trailing vortex system has to be calculated first. Use"
                             f" {response[vortex_system_type][1]} first.")
        return None

    def _assert_bound(self, vortex_system_type):
        response = {
            "blade": [self.coordinates_blade_bound, "rotor_bound(), blade_bound(), rotor(), or blade()"],
            "rotor": [self.coordinates_rotor_bound, "rotor_bound() or rotor()"]
        }
        if response[vortex_system_type][0] is None:
            raise ValueError(f"A {vortex_system_type} bound vortex system has to be calculated first. Use"
                             f" {response[vortex_system_type][1]} first.")
        return None

    def _combine_elementwise(self, called_from, coordinates_from, if_not_do, combine_to: str) -> None:
        """
        Combines element-wise coordinates into a np.ndarray of size (N,3).
        :param called_from:         Which function calls _combine_elementwise(). This is for user error clarification.
        :param coordinates_from:    The element-wise coordinates of the vortex system that are to be combined
        :param if_not_do:           Function that calculates these elementwise coordinates if they are not yet
                                    calculated
        :param save_to:             Class property name as string (without self.) to which the combined coordinates are
                                    saved.  :return: None
        """
        self._assert_properties(called_from) # assert that the rotor and wake properties have been set
        if coordinates_from is None: # if the element-wise coordinates do not yet exist
            coordinates_from = if_not_do() # then calculate them
        self.__dict__[combine_to] = np.asarray([ # save combined coordinates to a np.ndarray of size (N,3)
            [coord for coords in coordinates_from["x"].values() for coord in coords], # this appends all values from
            [coord for coords in coordinates_from["y"].values() for coord in coords], # the second layer dictionaries
            [coord for coords in coordinates_from["z"].values() for coord in coords], # to one another (axis-wise).
        ]).T
        return None

    def _rotate_combined(self,
                         called_from,
                         coordinates_from,
                         if_not_do,
                         rotate_to: str) -> None:
        """
        Creates a specific (bound or trailing) vortex system for the specified rotor. 

        It does that by taking the
        specific vortex system from a single blade and rotating it n_blades-1 times. The resulting vortex system is
        saved as a list with the vortex system (size (N,3)) of each blade as a unique entry.

        So
        - the list has so many entries as there are blades
        - every entry represents one vortex line for a single blade
        - columns: X, Y, Z
        - rows: points along the span
        :param called_from:         Which function calls _rotate_combined(). This is for user error clarification.
        :param coordinates_from:    The coordinates of the vortex system that is to be rotated
        :param if_not_do:           Function that calculates the combined coordinates of the vortex system if they are
                                    not yet calculated
        :param save_to:             Class property name as string (without self.) to which the combined coordinates are
                                    saved.
        :return:
        """
        self._assert_properties(called_from) # assert that the rotor and wake properties have been set
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

    @staticmethod
    def _float_to_ndarray(length: int, *args) -> list[np.ndarray]:
        """
        Loops through the input arguments to ensure they are numpy arrays. If a float is given, it is turned into an
        array of length 'length' with all values being the initial float.
        """
        return [arg if type(arg) == np.ndarray else np.asarray([arg for _ in range(length)]) for arg in args]
