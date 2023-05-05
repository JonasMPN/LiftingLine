# Script to create the wake positions of trailing vortices

import numpy as np
import matplotlib.pyplot as plt

class FrozenWake:
    def __init__(self):
        # rotor properties
        self.r_elements = None
        self.c_elements = None
        self.distance_control_point = None
        self.blade_rotation = None
        self.rotor_rotation_speed = None
        self.n_blades = None

        self.time_resolution: None

        self.wake_blade_elementwise = None
        self.wake_blade = None
        self.wake_rotor = None

    def set_rotor(self,
                  r_elements: np.ndarray,
                  c_elements: float or np.ndarray,
                  blade_rotation: float or np.ndarray,
                  rotor_rotation_speed: float,
                  n_blades: int = 3,
                  ) -> None:
        """
        The rotor coordinate system is such that the x faces downwind along the rotor axis, y faces to the left
        facing the rotor from the front and z point upwards. z goes along the leading edge of a blade that is point
        upward. Rotating a blade positively will turn the trailing edge to the left (when facing the rotor from the
        front).
        :param r_elements:              blade section radial position
        :param c_elements:              blade section chord length 
        :param blade_rotation:          rotation of the blade sections from the in plane position in Radian
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
        return None

    def blade(self, wake_speed: float, wake_length: float, time_resolution: int) -> None:
        """
        Creates the wake from one blade as one data structure. It does that by combining the points from the element
        wise wake to create a np.ndarray with size (len(r_elements), 3). All points are appended row wise.
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
        """
        if self.wake_blade_elementwise is None:
            self._blade_elementwise(wake_speed, wake_length, time_resolution)
        wake_x, wake_y, wake_z = list(), list(), list()
        for r in self.r_elements:
            wake_x += self.wake_blade_elementwise["x"][r]
            wake_y += self.wake_blade_elementwise["y"][r]
            wake_z += self.wake_blade_elementwise["z"][r]
        self.wake_blade = np.asarray([wake_x, wake_y, wake_z]).T
        return None

    def rotor(self, wake_speed: float, wake_length: float, time_resolution: int) -> None:
        """
        Creates the full wake. It does that by calculating the wake for a single blade and rotating that one
        n_blades-1 times. Returns a list with the wakes of each blade.
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
        """
        if self.wake_blade is None:
            self.blade(wake_speed, wake_length, time_resolution)
        self.wake_rotor = [self.wake_blade]
        for rot_angle in np.linspace(2*np.pi/self.n_blades, 2*np.pi*(1-1/self.n_blades), self.n_blades-1):
            rot_matrix = np.array([[1, 0, 0],
                                   [0, np.cos(rot_angle), -np.sin(rot_angle)],
                                   [0, np.sin(rot_angle), np.cos(rot_angle)]])
            self.wake_rotor.append(np.dot(self.wake_blade, rot_matrix))
        return None

    def blade_elementwise_visualisation(self) -> None:
        """
        Visualises the wake of one blade. Colours the trailing vortex of each blade element separately.
        :return: None
        """
        if self.wake_blade_elementwise is None:
            raise ValueError("A blade wake has to be calculated first. Use FrozenWake.blade() or FrozenWake.rotor() "
                             "first.")
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
        if self.wake_rotor is None:
            raise ValueError("A rotor wake has to be calculated first. Use FrozenWake.rotor() first.")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for wake, c in zip(self.wake_rotor, ["b", "g", "r", "c", "m", "y", "k"][:len(self.wake_rotor)]):
            for element in range(len(self.r_elements)):
                ax.plot(wake[element*(self.time_resolution+2):(element+1)*(self.time_resolution+2), 0],
                        wake[element*(self.time_resolution+2):(element+1)*(self.time_resolution+2), 1],
                        wake[element*(self.time_resolution+2):(element+1)*(self.time_resolution+2), 2], color=c)
        plt.show()
        return None


    def _blade_elementwise(self, wake_speed: float, wake_length: float, time_resolution: int) -> None:
        """
        Returns three dictionaries, one for each coordinate. Each dictionary is divided into the single blade elements.
        :param wake_speed:
        :param wake_length:
        :param time_resolution:
        :return:
        """
        self.time_resolution = time_resolution
        qc_elements = self.c_elements/4 # quarter chord
        x_qc = np.sin(self.blade_rotation)*qc_elements # x position of the quarter chord
        y_qc = np.cos(self.blade_rotation)*qc_elements # y position of the quarter chord
        x_c = np.sin(self.blade_rotation)*self.c_elements # x position of the trailing edge
        y_c = np.cos(self.blade_rotation)*self.c_elements # y position of the trailing edge
        # the swept vortices start a quarter chord behind the airfoil
        x_swept_trailing_vortices_start = x_c+x_qc # x position of the first node of the first swept trailing vortex
        y_swept_trailing_vortices_start = y_c+y_qc # y position of the first node of the first swept trailing vortex
        # the following three lines define the containers for the trailing vortex structure of each blade element
        # they are initialised with the fixed trailing vortex (first trailing vortex = fixed trailing vortex)
        x = {r: [x_qc[i], x_swept_trailing_vortices_start[i]] for i, r in enumerate(self.r_elements)}
        y = {r: [y_qc[i], y_swept_trailing_vortices_start[i]] for i, r in enumerate(self.r_elements)}
        z = {r: [r, r] for r in self.r_elements}
        for t in np.linspace(wake_length/(wake_speed*time_resolution), wake_length/wake_speed, time_resolution-1):
            angle = self.rotor_rotation_speed*t # increase the rotational angle
            for i, r in enumerate(self.r_elements):
                x[r].append(x_swept_trailing_vortices_start[i]+wake_speed*t) # pure convection
                # the first trailing vortex is parallel to the chord
                y[r].append(np.cos(angle)*y_swept_trailing_vortices_start[i]+r*np.sin(angle)) # rotated
                z[r].append(-np.sin(angle)*y_swept_trailing_vortices_start[i]+r*np.cos(angle)) # rotated
        self.wake_blade_elementwise = {"x": x, "y": y, "z": z}
        return None

    def _float_to_ndarray(self, *args) -> list[np.ndarray]:
        return [arg if type(arg) == np.ndarray else np.asarray([arg for _ in range(self.r_elements.size)]) for arg in
                args]

