# Task 11 - k :
# Operational point of the airfoils


# Idea:
# Plot polars 
# Plot aoa over the span for tsr 6 and 10 
# Discuss where out of range of optimum clcd 
# Discuss when we have most power and see whether there is correlation to the optimum working point 


from BEM import BEM
import numpy as np
import pandas as pd
from helper_functions import Helper
import matplotlib.pyplot as plt

helper = Helper()


####################### Polars

# read in airfoil data

def plot_polar(df):
    """
    Plot the polars of the given dataframe 
    """
    fig, axs = plt.subplots(3, 1, figsize=[6, 6])

    # CL over alpha
    axs[0].plot(df.alpha, df.cl)
    axs[1].plot(df.alpha, df.cd)
    axs[2].plot(df.alpha, df.cl / df.cd)

    axs[0].grid()
    axs[0].set_xlabel(r"$\alpha$ ($^\circ$)")
    axs[0].set_ylabel(r"$C_l$ (-)")

    axs[1].grid()
    axs[1].set_xlabel(r"$\alpha$ ($^\circ$)")
    axs[1].set_ylabel(r"$C_d$ (-)")

    axs[2].grid()
    axs[2].set_xlabel(r"$\alpha$ ($^\circ$)")
    axs[2].set_ylabel("$C_l / C_d$ (-)")

    maxid = df.idxmax()

    axs[0].scatter(df.alpha[maxid.clcd_ratio], df.cl[maxid.clcd_ratio], color="red")
    axs[1].scatter(df.alpha[maxid.clcd_ratio], df.cd[maxid.clcd_ratio], color="red")
    axs[2].scatter(df.alpha[maxid.clcd_ratio], df.clcd_ratio[maxid.clcd_ratio], color="red")

    plt.show()
    fig.savefig("../results/t11_polars.png", bbox_inches="tight")
    breakpoint()


def task11():
    ## Plot Polar Data & Find optimal AoA
    polar_file = "../data/polar.xlsx"
    df_polar = pd.read_excel(polar_file, skiprows=3)
    # bem = BEM(data_root="../data",
    #          file_airfoil="polar.xlsx")
    # Calculate optimal AoA
    df_polar["clcd_ratio"] = df_polar.cl / df_polar.cd
    AoA_opt = df_polar.alpha.loc[df_polar.clcd_ratio.idxmax()]
    print(f'AoA optimal:{AoA_opt}')
    plot_polar(df_polar)

    ## Plot actual data from BEM results
    BEMresults_file = '../data/BEM_results.dat'
    df_BEM = pd.read_csv(BEMresults_file, delimiter=',')
    check_TSR = 6

    tsr_list = [6, 8, 10]
    fig, ax = plt.subplots(1, 3, figsize=[14, 6])
    for tsr in tsr_list:
        df_iter = df_BEM.loc[(df_BEM['tsr'] == tsr) & ~(df_BEM['end_correction'] == 1)]

        # CHECK TSR = 6 Conditions : Start
        if tsr == check_TSR:
            print(f'\nChecking operational conditions @ tsr={tsr}')

            # Find index of max AoA for tsr = 6
            AoA_max_idx = df_iter['alpha'].idxmax()

            # Find max AoA for tsr = 6
            AoA_max = df_iter['alpha'].loc[AoA_max_idx]
            print(f'Max AoA : {AoA_max} deg')

            # Find the cl/cd ratio at max AoA
            test = df_iter.cl / df_iter.cd
            clcd_AoA_max = test.loc[AoA_max_idx]
            print(f'cl/cd @ max AoA {clcd_AoA_max}')
            print('Check done\n')
        # CHECK TSR = 6 Conditions : End

        ax[0].plot(df_iter['r_centre'] / 50, df_iter['alpha'], label=f'tsr:{tsr}')
        ax[1].plot(df_iter['r_centre'] / 50, df_iter['cl'] / df_iter['cd'], label=f'tsr:{tsr}')
        ax[2].plot(3 * (1-df_iter['r_centre']/50) + 1, df_iter['cl'], label=f'tsr:{tsr}')

    ax[0].plot(df_iter['r_centre'] / 50, df_iter.shape[0] * [AoA_opt], '--', label='AoA opt.')
    ax[0].set_title(r'AoA distribution along the blade')
    ax[0].set_xlabel(r'$r/R$ (-)')
    ax[0].set_ylabel(r'$AoA$ (deg)')
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title(r'Lift to Drag ratio along the blade')
    ax[1].set_xlabel(r'$r/R$ (-)')
    ax[1].set_ylabel(r'$c_l/c_d$ (-)')
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title(r'Lift vs Chord')
    ax[2].set_xlabel(r'Chord (m)')
    ax[2].set_ylabel(r'$c_l$ (-)')
    ax[2].grid()
    ax[2].legend()
    plt.show()

    fig.savefig("../results/t11.png", bbox_inches="tight")
    # breakpoint()

    print("Task 11 - Done")

if __name__ == "__main__":
    task11()
