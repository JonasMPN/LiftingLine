# Task 0: calculate the Reynolds number over the span

from BEM import BEM
import numpy as np
import matplotlib.pyplot as plt

bem = BEM(data_root="../data", file_airfoil="polar.xlsx")

def get_reynolds(U,L,nu):
    """
    Reynolds number calculation
    """
    return np.divide(U*L,nu) 

def get_velocity(r,rmax,tsr,u_0):
    """
    Calculate the absolute velocity array
    """
    u_rot = r * (tsr/rmax) * u_0 
    u = np.sqrt(np.multiply(u_rot,u_rot) + u_0**2) # absolute velocity
    return u 

def get_chord(r, r_max):
    """
        function to calculate chord along the span in m
        r: radial position along the blade in [m]
        r_max: maximum radius of the blade in [m]
    """
    return 3*(1-r/r_max)+1

def task0(tsr):
    fig, axs = plt.subplots(3,1, sharex=True) 
    linestyle=['-.',':',"--"]
    for tsr in tsr:
        bem_results = bem.get_results(wind_speed=10, tip_speed_ratio=tsr, pitch=-2)
        U = bem_results["inflow_speed"].to_numpy()
        radii = bem_results["r_centre"].to_numpy()
        nu = 1.5 * 10**-5
        inner_radius = 10
        outer_radius = 50
        # tsr_range = [6,8,10]
        #chords = [1] * len(radii) # np.ones((len(radii),1),) #
        chords = [get_chord(r,outer_radius) for r in radii]
        Re = get_reynolds(U, chords, nu)
    
      
        axs[0].plot(radii/outer_radius, chords, label = f"$\lambda$ = {tsr}")
        axs[1].plot(radii/outer_radius, U)
        axs[2].plot(radii/outer_radius, Re)
        
        axs[2].set_xlabel(r"$\mu (-)$")
        axs[0].set_ylabel("Chord (m)")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[2].set_ylabel("Reynolds number (-)")
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[0].yaxis.set_label_coords(-0.1,0.5)
        axs[1].yaxis.set_label_coords(-0.1,0.5)
        axs[2].yaxis.set_label_coords(-0.1,0.5)
        #plt.show()
    fig.legend(bbox_to_anchor=[1.1, 0.57])
    fig.savefig("../results/reynolds_number.png",bbox_inches="tight")

    

    




if __name__ =="__main__": 
    task0()
