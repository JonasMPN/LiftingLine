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
                    visualise_wake: bool=False) -> list[np.ndarray, np.ndarray , np.ndarray]:
    """
    Blade_rotation = pi means the chord is parallel to inflow. Blade rotation = 0 means chord is normal to inflow.
    :param r_elements:
    :param c_elements:
    :param wake_speed:
    :param wake_length:
    :param time_resolution:
    :param blade_rotation:
    :param rotor_rotation_speed:
    :param n_blades:
    :return:
    """
    blade_rotation = blade_rotation if type(blade_rotation) == list else [blade_rotation for _ in range(len(r_elements))]
    qc_elements = c_elements/4 if type(c_elements) == np.ndarray else np.asarray(c_elements)/4
    x_c = np.sin(blade_rotation)*c_elements
    y_c = np.cos(blade_rotation)*c_elements
    x_qc = np.sin(blade_rotation)*qc_elements
    y_qc = np.cos(blade_rotation)*qc_elements
    x_trailing_vortices_start = x_c+x_qc
    y_trailing_vortices_start = y_c+y_qc
    x = {r: [x_qc[i], x_trailing_vortices_start[i]] for i, r in enumerate(r_elements)}
    y = {r: [y_qc[i], y_trailing_vortices_start[i]] for i, r in enumerate(r_elements)}
    z = {r: [r, r] for r in r_elements}
    for t in np.linspace(0, wake_length/wake_speed, time_resolution):
        for i, r in enumerate(r_elements):
            x[r].append(x_trailing_vortices_start[i]+wake_speed*t)
            y[r].append(y_trailing_vortices_start[i]+r*np.sin(rotor_rotation_speed*t))
            z[r].append(r*np.cos(rotor_rotation_speed*t))

    wake_x, wake_y, wake_z = list(), list(), list()
    for r in r_elements:
        wake_x += x[r]
        wake_y += y[r]
        wake_z += z[r]
    wake_points = np.asarray([wake_x, wake_y, wake_z]).T

    if visualise_wake:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for r in r_elements: # or choose individual elements
            ax.plot(x[r], y[r], z[r])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.view_init(90, 90, -90)
        plt.show()


distance_matrix(r_elements=0.2+np.linspace(0,1,10), c_elements=np.linspace(0,0.2,10)[::-1],wake_speed=0.5,
                wake_length=10, time_resolution= 40, blade_rotation=0*np.pi/2, rotor_rotation_speed=np.pi/4,
                visualise_wake=True)