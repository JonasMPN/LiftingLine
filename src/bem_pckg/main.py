from BEM import BEM
import numpy as np
from helper_functions import Helper
import matplotlib.pyplot as plt
from task5 import task5
helper = Helper()
from task0 import task0
from task9 import task9

# Choose which parts of the code to run
do = {
    "different_tsr": True,
    "plots": False,
    "c": False,
    "task5": False,
    "test": False,
    "task0": False,
    "task9": False
}

bem = BEM(data_root="../data",
          file_airfoil="polar.xlsx")

if do["different_tsr"]:
    # Parameters
    bem.set_constants(rotor_radius=50, root_radius=50*0.2, n_blades=3, air_density=1.225)
    # Calculation
    for tsr in np.linspace(6,10,41):
        bem.solve_TUD(wind_speed=10, tip_speed_ratio=tsr, pitch=-2)

if do["task5"]:
    task5()

if do["plots"]:
    pass

if do["c"]:
    pass

if do["test"]:
    # This is a test/ example for using the bem function and using the outputs 
    # the object property "current_results" is a pandas dataframe with the results from the last / current computation
    bem_test = bem.get_results(wind_speed=10, tip_speed_ratio=8, pitch=-2)
    fig, axs = plt.subplots(4,1)
    axs[0].plot(bem_test.current_results.r_inner, bem_test.current_results.a)
    axs[1].plot(bem_test.current_results.r_inner, bem_test.current_results.a_prime)
    axs[2].plot(bem_test.current_results.r_inner, bem_test.current_results.alpha)
    axs[3].plot(bem_test.current_results.r_inner, bem_test.current_results.end_correction)
    
    ##### Make plot look a bit nicer
    # the following three lines do the same (and slightly more) as the commented lines below them.
    helper.handle_axis(axs, x_label="Radial position of the blade element centres (m)", grid=True,
                       y_label=["Induction (-)", "Induction (-)", r"$\alpha$", "Blade end correction loss (-)"])
    helper.handle_figure(fig, size=(5,7), show=True)
    # axs[0].set_xlabel("Radial position of the center point[m]")
    # axs[1].set_xlabel("Radial position of the center point[m]")
    # axs[2].set_xlabel("Radial position of the center point[m]")
    # axs[3].set_xlabel("Radial position of the center point[m]")

    # axs[0].set_ylabel("Induction []")
    # axs[1].set_ylabel("Induction []")
    # axs[2].set_ylabel(r"$\alpha$")
    # axs[3].set_ylabel("Tip loss factor []")
    # plt.show()
    print("Done testing")

if do["task0"]:
    tsr = [6,8,10]
    task0(tsr)
if do["task9"]:
    tsr = [6,8,10]
    task9(tsr)

