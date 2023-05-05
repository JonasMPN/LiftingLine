# Script to create the wake positions of trailing vortices

import numpy as np
import matplotlib.pyplot as plt


def distance_matrix(r_elements: list or np.ndarray,
                    c_elements: list or np.ndarray,
                    wake_speed: float,
                    wake_length: float,
                    time_resolution: int,
                    blade_rotation: float or list,
                    rotor_rotation_speed: float,
                    n_blades: int=3,
                    visualise_blade_wake: bool=False,
                    visualise_wake: bool=False) -> list[np.ndarray, np.ndarray , np.ndarray]:
    """
    Create the matrix containing all distances 

    Blade_rotation = pi means the chord is parallel to inflow. Blade rotation = 0 means chord is normal to inflow.
    
    :param r_elements:              blade section radial position
    :param c_elements:              blade section chord length 
    :param wake_speed:              Convection speed of the wake
    :param wake_length:             Length of the resolved wake in the downstream direction
    :param time_resolution:         time resolution
    :param blade_rotation:          rotation of the blade sections from the in plane position in Radian
    :param rotor_rotation_speed:    Rotational speed of the rotor
    :param n_blades:                Number of blades
    :return:                        Function does not return  anything
    """

    # checks if a list of blade rotations is given. if not, create a list with the uniform value 
    blade_rotation = blade_rotation if type(blade_rotation) == list else [blade_rotation for _ in range(len(r_elements))]
    # create a nd array that has stored 1/4 of c. c is probably the chord?
    qc_elements = c_elements/4 if type(c_elements) == np.ndarray else np.asarray(c_elements)/4
    # create x and y coponents of the  
    x_c = np.sin(blade_rotation)*c_elements # streamwise position of the trailing edge
    y_c = np.cos(blade_rotation)*c_elements # radial position of the trailing edge
    
    # create x and y component of the quarter chord point in a similar manner 
    x_qc = np.sin(blade_rotation)*qc_elements
    y_qc = np.cos(blade_rotation)*qc_elements

    # define x and y position where the trailing vortices would start as the addition of one chord length and 1/4 of it
    x_trailing_vortices_start = x_c+x_qc
    y_trailing_vortices_start = y_c+y_qc

    # create dictionaries with the radial position as index containing pairs of quarter chord position and trailing vortex start position

    x = {r: [x_qc[i], x_trailing_vortices_start[i]] for i, r in enumerate(r_elements)}
    y = {r: [y_qc[i], y_trailing_vortices_start[i]] for i, r in enumerate(r_elements)}
    z = {r: [r, r] for r in r_elements} # and the z position is of course the radius, as the chord for a section is normal to the z direction
    # compute a time based on the wake dimensions that offsets a position in the wake from the rotor plane
    for t in np.linspace(0, wake_length/wake_speed, time_resolution): # if t starts at 0, we double the position of the trailing edge
        angle = rotor_rotation_speed*t # angle which the position of the wake is rotated to the rotor plane
        # Q: shouldnt this be negative, as we go backwards in time?
         
        # Add the positions of the rotated wake to the dictionaries  
        for i, r in enumerate(r_elements):
            x[r].append(x_trailing_vortices_start[i]+wake_speed*t)
            y[r].append(np.cos(angle)*y_trailing_vortices_start[i]+r*np.sin(angle))
            z[r].append(-np.sin(angle)*y_trailing_vortices_start[i]+r*np.cos(angle))
    
    # Do nice plots
    if visualise_blade_wake:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for r in r_elements: # or choose individual elements
            ax.plot(x[r], y[r], z[r])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.view_init(90, 90, -90)
        plt.show()
    
    # Put all the wake positions for all radii in a single list 
    # Maybe a np nd array could have alleviated the problem?
    wake_x, wake_y, wake_z = list(), list(), list()
    for r in r_elements:
        wake_x += x[r]
        wake_y += y[r]
        wake_z += z[r]
    
    # Aaahh I see, it is all now put into a nd array
    base_wake = np.asarray([wake_x, wake_y, wake_z]).T
    wakes = [base_wake] # put the wake geometry array into a list
    
    # creates the wakes for the other blades by rotating the first wake
    for rot_angle in np.linspace(2*np.pi/n_blades, 2*np.pi*(1-1/n_blades), n_blades-1):
        rot_matrix = np.array([[1, 0, 0],
                               [0, np.cos(rot_angle), -np.sin(rot_angle)],
                               [0, np.sin(rot_angle), np.cos(rot_angle)]])
        wakes.append(np.dot(base_wake, rot_matrix)) # and puts all new wakes in the wakes list
    
    # more nice plots
    if visualise_wake:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        colour = ["b", "g", "k"]
        for wake, c in zip(wakes, colour):
            for element in range(len(r_elements)):
                ax.plot(wake[element*(time_resolution+2):(element+1)*(time_resolution+2), 0],
                        wake[element*(time_resolution+2):(element+1)*(time_resolution+2), 1],
                        wake[element*(time_resolution+2):(element+1)*(time_resolution+2), 2], color=c)
        plt.show()


# run the function
if __name__ == "__main__":
    distance_matrix(r_elements=0.2+np.linspace(0,1,5), c_elements=np.linspace(0,0.2,5)[::-1],wake_speed=0.5,
                    wake_length=5, time_resolution=50, blade_rotation=0*np.pi/2, rotor_rotation_speed=np.pi/4,
                    visualise_blade_wake=False, visualise_wake=True)
