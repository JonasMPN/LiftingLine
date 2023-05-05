#### Optimizing CP 


# We optimise TSR 6, as here the balde obviously stalls. 
# We optimize Chord and twist 

# Kutta Joukowsky 


from BEM import BEM
import numpy as np
from helper_functions import Helper
import matplotlib.pyplot as plt
from task5 import task5
import matplotlib.pyplot as plt
helper = Helper()

# Choose whicht parts of the code to run 

def get_twist(r, r_max):
    """
        function to get the twist along the blade in radians
        r: radial position along the blade in [m]
        r_max: maximum radius of the blade in [m]
        out: twist in radians
    """
    return np.radians(14*(1-r/r_max))

def get_chord(r, r_max):
    """
        function to calculate chord along the span in m
        r: radial position along the blade in [m]
        r_max: maximum radius of the blade in [m]
    """
    return 3*(1-r/r_max)+1
def task7():

    bem = BEM(data_root="../data",
              file_airfoil="polar.xlsx")

    bem.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)
    bem.optimize_TUD2(wind_speed=10, tip_speed_ratio=6, pitch=-2)
    
def task7_plot():
    """
    Plot the twist and chord distribution based on data stored in files from previous calculations
    """
    resolution = 200
    r = 50
    radii = np.linspace(0.2,1,199) # This is not exact! -> but marginal differences, so it is good enough to show the approach
    radii_dim = np.linspace(20,50,199) # This is not exact! -> but marginal differences, so it is good enough to show the approach

    twist_list = [get_twist(r_current, r) for r_current in radii_dim]  # get the reference twist
    chord_list = [get_chord(r_current, r)/r for r_current in radii_dim]  # get the reference chord

    chord = np.loadtxt("../results/chord.txt") /r   #  get chord from file non- dimensionalize
    twist = np.loadtxt("../results/twist.txt")      # get twist from fil
    fig, axs = plt.subplots(2,1, figsize=(6,4))

    axs[0].plot(radii, chord, label="optimised") 
    axs[1].plot(radii, np.rad2deg(twist), label="optimised") 

    axs[0].plot(radii, chord_list, label="reference design") 
    axs[1].plot(radii, np.rad2deg(twist_list), label="reference design") 
    
    axs[0].grid()
    #axs[0].set_xlabel("$\mu$ (-)")
    axs[0].set_ylabel("c/R (-)")
    axs[0].legend()

    axs[1].grid()
    axs[1].set_xlabel("$\mu$ (-)")
    axs[1].set_ylabel("twist $(^\circ)$")
    axs[1].legend()
    plt.savefig("../results/optimum_chordtwist.png",bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    task7_plot()
