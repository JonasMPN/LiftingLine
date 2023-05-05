# Tip loss correction stuff for task 5

from BEM import BEM
import numpy as np
from helper_functions import Helper
import matplotlib.pyplot as plt
helper = Helper()

# Choose which parts of the code to run 
def task5():
    print("Task 5: Start")

    ### Create a bem object for every calculation
    bem_tsr6 = BEM(data_root="../data",
              file_airfoil="polar.xlsx")

    bem_tsr8 = BEM(data_root="../data",
              file_airfoil="polar.xlsx")
    bem_tsr10 = BEM(data_root="../data",
              file_airfoil="polar.xlsx")
    
    # Bem object for calc without tip loss 
    bem_tsr8_no_tip_loss = BEM(data_root="../data",
              file_airfoil="polar.xlsx")
    # Parameters
    bem_tsr6.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)
    bem_tsr8.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)
    bem_tsr10.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)

    bem_tsr8_no_tip_loss.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)

    ##### Calculation at 3 differnt tsrs
    resolution = 200
    bem_tsr6.solve_TUD(wind_speed=10, tip_speed_ratio=6, pitch=-2, resolution=resolution)
    bem_tsr8.solve_TUD(wind_speed=10, tip_speed_ratio=8, pitch=-2 , resolution=resolution)
    bem_tsr10.solve_TUD(wind_speed=10, tip_speed_ratio=10, pitch=-2, resolution =resolution)


    bem_tsr8_no_tip_loss.solve_TUD(wind_speed=10, tip_speed_ratio=8, pitch=-2 , resolution=resolution,tip_loss_correction=False, root_loss_correction = False)
    #### Grab data that we want



    #### Plots


    ## Plot for the tip loss correction factors
    fig, axs = plt.subplots(1,1, figsize=[6,4])
    axs.plot(bem_tsr6.current_results.r_centre/50,bem_tsr6.current_results.end_correction,label ="Tsr = 6")
    axs.plot(bem_tsr8.current_results.r_centre/50,bem_tsr8.current_results.end_correction,label ="Tsr = 8")
    axs.plot(bem_tsr10.current_results.r_centre/50,bem_tsr10.current_results.end_correction,label ="Tsr = 10")
    axs.legend()
    axs.set_xlabel("radial position $\mu$ (-)")
    axs.set_ylabel("tip and root loss correction factor $f$ (-)")
    axs.grid()
    plt.tight_layout()
    #plt.show()
    fig.savefig("../results/tip_root_loss.png",bbox_inches="tight")


    # Plot to show the difference of adding the tip loss
    fig2, axs2 = plt.subplots(1,1, figsize=[6,4])
    
    axs2.plot(bem_tsr8.current_results.r_centre/50,bem_tsr8.current_results.a,label ="prandtl corrected")
    axs2.plot(bem_tsr8_no_tip_loss.current_results.r_centre/50,bem_tsr8_no_tip_loss.current_results.a,label ="no correction")
    
    axs2.legend()
    axs2.set_xlabel("radial position $\mu$ (-)")
    axs2.set_ylabel(r"axial inducton factor $a$ (-)")
    axs2.grid()
    plt.tight_layout()
    fig2.savefig("../results/tip_root_loss_induction.png",bbox_inches="tight")

    # Loads

    # Plot to show the difference of adding the tip loss
    fig3, axs3 = plt.subplots(1,2, figsize=[6,4])
    
    axs3[0].plot(bem_tsr8.current_results.r_centre/50,bem_tsr8.current_results.f_t,label ="prandtl corrected")
    axs3[0].plot(bem_tsr8_no_tip_loss.current_results.r_centre/50,bem_tsr8_no_tip_loss.current_results.f_t,label ="no correction")
    axs3[1].plot(bem_tsr8.current_results.r_centre/50,bem_tsr8.current_results.f_n,label ="prandtl corrected")
    axs3[1].plot(bem_tsr8_no_tip_loss.current_results.r_centre/50,bem_tsr8_no_tip_loss.current_results.f_n,label ="no correction")
    axs3[0].legend()
    axs3[1].legend()
    axs3[0].set_xlabel("radial position $\mu$ (-)")
    axs3[0].set_ylabel(r"tangential loads (N/m)")
    axs3[1].set_xlabel("radial position $\mu$ (-)")
    axs3[1].set_ylabel(r"normal loads  $(N/m)$")
    axs3[0].grid()
    axs3[1].grid()
    plt.tight_layout()

    fig3.savefig("../results/task5_loads.png",bbox_inches="tight")


    # Plot to show the difference of adding the tip loss
    fig4, axs4 = plt.subplots(1,1, figsize=[6,4])
    
    axs4.plot(bem_tsr8.current_results.r_centre/50,bem_tsr8.current_results.alpha,label ="prandtl corrected")
    axs4.plot(bem_tsr8_no_tip_loss.current_results.r_centre/50,bem_tsr8_no_tip_loss.current_results.alpha,label ="no correction")
    
    axs4.legend()
    axs4.set_xlabel("radial position $\mu$ (-)")
    axs4.set_ylabel(r"angle of attack $\alpha$ ($^\circ$)")
    axs4.grid()
    plt.tight_layout()
    fig4.savefig("../results/task5_tip_root_loss_aoa.png",bbox_inches="tight")
    plt.show()
    # Final stuff
    print("Task 5: Done")

if __name__ == "__main__":
    task5()

