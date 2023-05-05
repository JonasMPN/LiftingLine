# Script to create the wake positions of trailing vortices
import numpy as np
import matplotlib.pyplot as plt

class FrozenWake:
    def __init__(self):
        # rotor structural properties
        self.r_elements = None
        self.c_elements = None
        self.blade_rotation = None
        self.rotor_rotation_speed = None
        self.n_blades = None

        # wake properties
        self.wake_speed = None
        self.wake_length = None
        self.time_resolution = None

        # control point
        self.distance_control_point = None

        # instance properties
        self.wake_blade_elementwise = None
        self.wake_blade = None
        self.wake_rotor = None

        # error clarification
        self.rotor_set = False
        self.wake_set = False

    def set_rotor(self,
                  r_elements: np.ndarray,
                  c_elements: float or np.ndarray,
                  blade_rotation: float or np.ndarray,
                  rotor_rotation_speed: float,
                  n_blades: int = 3
                  ) -> None:
        """
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
        # checks if a list of chord and blade rotations is given. if not, create a list with the uniform value
        self.c_elements, self.blade_rotation = self._float_to_ndarray(c_elements, blade_rotation)
        if self.r_elements.shape != self.c_elements.shape or self.r_elements.shape  != self.blade_rotation.shape:
            raise ValueError("'r_elements', 'c_elements', and 'blade_rotation' do not have the same length. This only "
                             "happens if 'c_elements' or 'blade_rotation' are specified as a list. The shapes are:"
                             f"r_elements: {self.r_elements.size}, c_elements: {self.c_elements.size}, blade_rotation:"
                             f" {self.blade_rotation.size}")
        self.rotor_rotation_speed = rotor_rotation_speed
        self.n_blades = n_blades
        # if a new rotor is defined, the old wakes are no longer correct and are therefore deleted
        self.wake_blade_elementwise = None
        self.wake_blade = None
        self.wake_rotor = None
        # error clarification for user
        self.rotor_set = True
        return None

    def set_wake_properties(self, wake_speed: float, wake_length: float, time_resolution: int) -> None:
        self._set(**{param: value for param, value in locals().items() if param != "self"})
        # error clarification for user
        self.wake_set = True
        return None

    def blade(self) -> None:
        """
        Creates the wake from one blade as one data structure. It does that by combining the points from the element
        wise wake to create a np.ndarray with size (len(r_elements), 3). All points are appended row wise.
        
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
        """
        self._assert_properties(self.blade)
        if self.wake_blade_elementwise is None:
            self._blade_elementwise()
        wake_x, wake_y, wake_z = list(), list(), list()
        for r in self.r_elements:
            wake_x += self.wake_blade_elementwise["x"][r]
            wake_y += self.wake_blade_elementwise["y"][r]
            wake_z += self.wake_blade_elementwise["z"][r]
        self.wake_blade = np.asarray([wake_x, wake_y, wake_z]).T
        return None

    def rotor(self) -> None:
        """
        Creates the full wake. It does that by calculating the wake for a single blade and rotating that one
        n_blades-1 times. Returns a list with the wakes of each blade.
        
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
        """
        self._assert_properties(self.rotor)
        if self.wake_blade is None:
            self.blade()
        self.wake_rotor = [self.wake_blade]
        for rot_angle in np.linspace(2*np.pi/self.n_blades, 2*np.pi*(1-1/self.n_blades), self.n_blades-1):
            rot_matrix = np.array([[1, 0, 0],
                                   [0, np.cos(rot_angle), np.sin(rot_angle)],
                                   [0, -np.sin(rot_angle), np.cos(rot_angle)]])
            self.wake_rotor.append(np.dot(self.wake_blade, rot_matrix))
        return None

    def induction_matrix(self, control_point: float) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the induction matrix from the wake of the rotor on all control points of one blade. The control
        point has to be specified as a multiple of the chord length. The distance that results is measured from the
        leading edge parallel to the chord.
        :param control_point:
        :return:
        """
        self._assert_wake("rotor") # a rotor wake has to exist
        x_control_points = -np.sin(self.blade_rotation)*self.c_elements*control_point # place control points
        y_control_points = np.cos(self.blade_rotation)*self.c_elements*control_point # place control points
        z_control_points = self.r_elements # place control points
        control_points = np.asarray([x_control_points, y_control_points, z_control_points]).T # collect all coordinates

        n_circulations = len(self.r_elements) # because at each blade element a vortex line is shed
        single_wake_induction_matrices = { # the inductions are calculated for individual wakes first (debugging help)
            "x": [np.zeros((n_circulations, n_circulations)) for _ in range(self.n_blades)],
            "y": [np.zeros((n_circulations, n_circulations)) for _ in range(self.n_blades)],
            "z": [np.zeros((n_circulations, n_circulations)) for _ in range(self.n_blades)]
        }
        for wake_i, wake in enumerate(self.wake_rotor): # iterate over the wakes of each blade
            for induced_element in range(len(self.r_elements)): # iterate over each blade element
                control_point = control_points[induced_element, :] # get the control point of the current blade element
                for inducing_element in range(len(self.r_elements)): # iterate over the trailing vortex of each blade
                    # element. Every blade element thus influences (thus inducing_element) the current element (which
                    # is therefore called induced_element, because it 'receives' an induced velocity.

                    # get the start points of each vortex element of the trailing vortex from the inducing blade element
                    vortex_starts = wake[inducing_element*(self.time_resolution+1):(inducing_element+1)*(self.time_resolution+1)-1]
                    # get the end points of each vortex element of the trailing vortex from the inducing blade element
                    vortex_ends = wake[1+inducing_element*(self.time_resolution+1):(inducing_element+1)*(self.time_resolution+1)]
                    induction_factors = np.zeros(3)
                    for vortex_start, vortex_end in zip(vortex_starts, vortex_ends): # iterate over all vortex elements
                        induction_factors += self._induction_factor(vortex_start, vortex_end, control_point)
                    # place the induction factors (x, y, and z component) of this wake in its respective wake
                    # induction matrix
                    single_wake_induction_matrices["x"][wake_i][induced_element, inducing_element] = induction_factors[0]
                    single_wake_induction_matrices["y"][wake_i][induced_element, inducing_element] = induction_factors[1]
                    single_wake_induction_matrices["z"][wake_i][induced_element, inducing_element] = induction_factors[2]

        induction_matrices = { # initialise container for the final, summed together induction factor matrices
            "x": np.zeros((n_circulations, n_circulations)),
            "y": np.zeros((n_circulations, n_circulations)),
            "z": np.zeros((n_circulations, n_circulations))
        }
        for direction in ["x", "y", "z"]:
            for induction_mat in single_wake_induction_matrices[direction]:
                induction_matrices[direction] += induction_mat # sum the contributions of all blade wakes
        return [induction_matrices["x"], induction_matrices["y"], induction_matrices["z"]]

    def blade_elementwise_visualisation(self) -> None:
        """
        Visualises the wake of one blade. Colours the trailing vortex of each blade element separately.
        
        :return: None
        """
        self._assert_wake("blade")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for r in self.r_elements:  # or choose individual elements
            ax.plot(self.wake_blade_elementwise["x"][r], self.wake_blade_elementwise["y"][r],
                    self.wake_blade_elementwise["z"][r])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.view_init(90, 90, -90)
        plt.show()
        return None

    def rotor_visualisation(self) -> None:
        """
        Visualises the wake of the whole rotor. Currently supports a maximum of 7 wakes (due to colouring).
        
        :return: None
        """
        self._assert_wake("rotor")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for wake, c in zip(self.wake_rotor, ["b", "g", "r", "c", "m", "y", "k"][:len(self.wake_rotor)]):
            for element in range(len(self.r_elements)):
                ax.plot(wake[element*(self.time_resolution+1):(element+1)*(self.time_resolution+1), 0],
                        wake[element*(self.time_resolution+1):(element+1)*(self.time_resolution+1), 1],
                        wake[element*(self.time_resolution+1):(element+1)*(self.time_resolution+1), 2], color=c)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
        return None

    def _blade_elementwise(self) -> None:
        """
        Returns three dictionaries, one for each coordinate. Each dictionary is divided into the single blade elements.
        
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
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
        for t in np.linspace(self.wake_length/(self.wake_speed*self.time_resolution),
                             self.wake_length/self.wake_speed,
                             self.time_resolution-1):
            angle = self.rotor_rotation_speed*t-np.pi/2 # It is assumed that the rotor is rotating clockwise when
            # looking downwind. The -np.pi/2 rotate the t=0 position of the blade ot be parallel to the z-axis.
            for i, r in enumerate(self.r_elements):
                x[r].append(x_swept_trailing_vortices_start[i]+self.wake_speed*t) # pure convection
                # the first trailing vortex is parallel to the chord
                y[r].append(-np.sin(angle)*y_swept_trailing_vortices_start[i]+r*np.cos(angle)) # rotated
                z[r].append(-np.cos(angle)*y_swept_trailing_vortices_start[i]-r*np.sin(angle)) # rotated
        self.wake_blade_elementwise = {"x": x, "y": y, "z": z}
        return None

    def _induction_factor(self,
                          vortex_start: np.ndarray,
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
        vec_R_1 = induction_point-vortex_start # vector from vortex_start to the induction point
        vec_R_2 = induction_point-vortex_end # vector from vortex_end to the induction point
        R_1 = np.linalg.norm(vec_R_1) # distance between the vortex start point and the induction point
        R_2 = np.linalg.norm(vec_R_2) # distance between the vortex end point and the induction point
        vec_plane_normal = np.cross(vec_R_1, vec_R_2) # vector that's normal on the plane spanned by the three points
        if not np.any(vec_plane_normal): # this happens when the induction point lies on the extended vortex line
            return np.zeros(3)
        l_sq_plane_normal = np.dot(vec_plane_normal, vec_plane_normal) # squared length of that plane normal vector
        vec_vortex = vortex_end-vortex_start # vector representing the vortex line
        fac_1 = np.dot(vec_vortex, vec_R_1) # zero clue what this does later
        fac_2 = np.dot(vec_vortex, vec_R_2) # zero clues what this does later
        K = 1/(4*np.pi*l_sq_plane_normal)*(fac_1/R_1-fac_2/R_2) # some magic factor
        return K*vec_plane_normal # boom done

    def _float_to_ndarray(self, *args) -> list[np.ndarray]:
        return [arg if type(arg) == np.ndarray else np.asarray([arg for _ in range(self.r_elements.size)]) for arg in
                args]

    def _set(self, **kwargs) -> None:
        """
        Sets parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
        :param kwargs:
        :return:
        """
        existing_parameters = [*self.__dict__]
        for parameter, value in kwargs.items():
            if parameter not in existing_parameters:
                raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
            self.__dict__[parameter] = value
        return None

    def _assert_properties(self, fnc):
        for params, was_set in {"rotor": self.rotor_set, "wake": self.wake_set}.items():
            if not was_set:
                raise ValueError(f"{params} properties have to be set before using {fnc.__name__}().")

    def _assert_wake(self, wake_type):
        response = {
            "blade": [self.wake_blade, "rotor() or blade()"],
            "rotor": [self.wake_rotor, "rotor()"]
        }
        if response[wake_type][0] is None:
            raise ValueError(f"A {wake_type} wake has to be calculated first. Use {response[wake_type][1]} first.")
