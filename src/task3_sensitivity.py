import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from bem_pckg.helper_functions import Helper  # problem -> different versions of these helper functions
from bem_pckg import BEM
from bem_pckg import twist_chord
from vortex_system import VortexSystem
from task1 import calc_ll, calc_forces, force_to_coeff, c_normal, c_tangent

"""
Do sensitivity analysis on the following parameters:

- Convection speed for the wake
- Blade discretization approach (uniform, cosine)
- Azimuthal discretization ????
- Length of the wake (number  of rotations)
- Vortex core radius (additional)
"""


def task3(wake_speed, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius):

    # Input operational data
    operational_data = {
        "v_0": 10,
        "air_density": 1.225,
        "tsr": 8,
        "radius": 50,
        "n_blades": 3,
        "inner_radius": 0.2 * 50,
        "pitch": np.deg2rad(-2),
        "resolution_ll": 10
    }

    # Loop over different parameters
    fig, ax = plt.subplots(3, 2, figsize=(6, 6))
    for w_s, b_d, a_d, w_l, v_c_r in zip(wake_speed, blade_discretization, azimuthal_discretization, wake_length,
                                         vortex_core_radius):
        # Assemble sensitivity parameters
        sensitivity_params = {
            "vortex_core_radius": v_c_r,
            "debug": False,
            "wake_length": w_l,
            "disctretization": b_d,
            "residual_max": 1e-10,
            "n_iter_max": 1000,
            "wake_speed": w_s,
            "resolution_wake": a_d
        }

        # Calculate for different parameters
        ll_results = calc_ll(**operational_data, **sensitivity_params)
        # Calculate loads
        ll_loads = calc_forces(ll_results["cl"], ll_results["cd"], ll_results["phi"], ll_results["chord"],
                               ll_results["element_length"], ll_results["inflow_speed"],
                               operational_data["air_density"],
                               operational_data["inner_radius"], operational_data["radius"],
                               operational_data["n_blades"])

        # Non-dimensional loads
        ll_cn = c_normal(ll_results["phi"], ll_results["cl"], ll_results["cd"])
        ll_ct = c_tangent(ll_results["phi"], ll_results["cl"], ll_results["cd"])


        # Plotting #
        # Axial induction a
        ax[0, 0].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['a'])
        ax[0, 0].set_ylabel('Axial induction (-)')
        ax[0, 0].grid(True)

        # Azimuthal induction a'
        ax[0, 1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['a_prime'])
        ax[0, 1].set_ylabel('Azimuthal induction (-)')
        ax[0, 1].grid(True)

        # Lift coefficient cl
        ax[1, 0].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_results['cl'])
        ax[1, 0].set_ylabel(r'$c_l$ (-)')
        ax[1, 0].grid(True)

        # Non dimensional bound circulation
        omega = operational_data["tsr"] * operational_data["v_0"] / operational_data["radius"]
        bound_circulation_nondim = ll_results['bound_circulation'] * operational_data["n_blades"] * omega / (np.pi * operational_data["v_0"]**2)
        ax[1, 1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), bound_circulation_nondim)
        ax[1, 1].set_ylabel(r'Bound circulation $\frac{\Gamma}{\pi V_{\infty}^2 / \omega B}$')
        ax[1, 1].grid(True)

        # Normal Loads
        ax[2, 0].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_loads['f_n'])
        # ax[0, 1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_cn)
        ax[2, 0].set_ylabel(r'Normal loads (N/m)')
        ax[2, 0].set_xlabel(r'Radial position $\mu$(-)')
        ax[2, 0].grid(True)

        # Tangential Loads
        ax[2, 1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_loads['f_t'])
        # ax[1, 1].plot(ll_results['r_centre'] / ll_results['r_centre'].max(), ll_ct)
        ax[2, 1].set_ylabel(r'Tangential loads (N/m)')
        ax[2, 1].set_xlabel(r'Radial position $\mu$(-)')
        ax[2, 1].grid(True)


    # Legend
    plt.tight_layout()
    if wake_speed.count(wake_speed[0]) != len(wake_speed):
        legend = [f'{x} m/s' for x in wake_speed]
        plt.legend(legend)
        plt.savefig("../results/task3/task3_wake_speed.pdf", bbox_inches='tight')
        plt.savefig("../results/task3/task3_wake_speed.pgf", bbox_inches='tight')
    elif blade_discretization.count(blade_discretization[0]) != len(blade_discretization):
        plt.legend(blade_discretization)
        plt.savefig("../results/task3/task3_blade_discr.pdf", bbox_inches='tight')
        plt.savefig("../results/task3/task3_blade_discr.pgf", bbox_inches='tight')
    elif azimuthal_discretization.count(azimuthal_discretization[0]) != len(azimuthal_discretization):
        legend = [f'{x} points' for x in azimuthal_discretization]
        plt.legend(legend)
        plt.savefig("../results/task3/task3_azimuth_discr.pdf", bbox_inches='tight')
        plt.savefig("../results/task3/task3_azimuth_discr.pgf", bbox_inches='tight')
    elif wake_length.count(wake_length[0]) != len(wake_length):
        legend = [f'{x/(2*operational_data["radius"])} D' for x in wake_length]
        plt.legend(legend)
        plt.savefig("../results/task3/task3_wake_length.pdf", bbox_inches='tight')
        plt.savefig("../results/task3/task3_wake_length.pgf", bbox_inches='tight')
    else:
        legend = [f'{x} m' for x in vortex_core_radius]
        plt.legend(legend)
        plt.savefig("../results/task3/task3_vortex_core.pdf", bbox_inches='tight')
        plt.savefig("../results/task3/task3_vortex_core.pgf", bbox_inches='tight')
    plt.show()

    return


if __name__ == '__main__':

    ####################################################################################################################
    # Sensitivity analysis
    ####################################################################################################################

    parameters = {
        "wakeSpeed": True,
        "bladeDiscr": True,
        "wakeDiscr": True,
        "wakeLength": True,
        "vortexCoreRad": True

    }

    if parameters["wakeSpeed"]:
        # Variable wake speeds
        wake_speed_range = [5, 10, 15]
        blade_discretization = len(wake_speed_range) * ['uniform']
        azimuthal_discretization = len(wake_speed_range) * [50]
        wake_length = len(wake_speed_range) * [1 * 2 * 50]
        vortex_core_radius = len(wake_speed_range) * [1]
        task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius)

    if parameters["bladeDiscr"]:
        # Variable blade discretization
        blade_discretization = ['uniform', 'sin']
        wake_speed_range = len(blade_discretization) * ["from_BEM"]
        azimuthal_discretization = len(blade_discretization) * [50]
        wake_length = len(blade_discretization) * [1 * 2 * 50]
        vortex_core_radius = len(blade_discretization) * [1]
        task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius)

    if parameters["wakeDiscr"]:
        # Variable azimuthal discretization
        azimuthal_discretization = [25, 50, 75, 100]
        wake_speed_range = len(azimuthal_discretization) * ["from_BEM"]
        blade_discretization = len(azimuthal_discretization) * ['uniform']
        wake_length = len(azimuthal_discretization) * [1 * 2 * 50]
        vortex_core_radius = len(azimuthal_discretization) * [1]
        task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius)

    if parameters["wakeLength"]:
        # Variable wake length
        diameter = 2 * 50
        wake_length = [1 * diameter, 3 * diameter, 5 * diameter]
        wake_speed_range = len(wake_length) * ["from_BEM"]
        blade_discretization = len(wake_length) * ['uniform']
        azimuthal_discretization = len(wake_length) * [50]
        vortex_core_radius = len(wake_length) * [1]
        task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius)

    if parameters["vortexCoreRad"]:
        # Variable vortex core radius
        vortex_core_radius = [1, 10, 20]
        wake_speed_range = len(vortex_core_radius) * ["from_BEM"]
        blade_discretization = len(vortex_core_radius) * ['uniform']
        azimuthal_discretization = len(vortex_core_radius) * [50]
        wake_length = len(vortex_core_radius) * [1 * 2 * 50]
        task3(wake_speed_range, blade_discretization, azimuthal_discretization, wake_length, vortex_core_radius)
