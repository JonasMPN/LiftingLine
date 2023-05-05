import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from BEM import BEM


def task9(tsr_list):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    
    # calculate induction from the forces to get the average induction over the stream disc.
    color = ["blue", "red", "green"]
    linestyle=['-.',':',"--"]
    
    bem = BEM(data_root="../data", file_airfoil="polar.xlsx")
    
    figa,axa = plt.subplots(len(tsr_list),3, sharex = "col", sharey=True, figsize=(14, 8))
    figb,axb = plt.subplots(len(tsr_list),3, sharex = "col", sharey=True, figsize=(14, 8))
    figc,axc = plt.subplots(len(tsr_list),3, sharex = "col", sharey=True, figsize=(14, 8))
    figd,axd = plt.subplots(len(tsr_list),3, sharex = "col", sharey=True, figsize=(14, 8))
    
    for n, tsr in enumerate(tsr_list):
        bem_results = bem.get_results(wind_speed=10, tip_speed_ratio=tsr, pitch=-2)
        
        
        
        rho = 1.225 # density of air [kg/m3]
        p_static_a = np.ones(len(bem_results))*101325 # static pressure (Point A)[N/m2]
        

        # convert to numpy
        v0 = bem_results.wind_speed.to_numpy()
        r_b = bem_results.r_centre.to_numpy().astype(float)
        r_outer = bem_results.r_outer.to_numpy()
        r_inner = bem_results.r_inner.to_numpy()
        a = bem_results.a.to_numpy()
        f_n = bem_results.f_n.to_numpy()
        B  = 3
        
        f_n = f_n * B * (r_outer-r_inner)/(np.pi*(r_outer**2-r_inner**2)) 
        # 
        f = bem_results.end_correction.to_numpy()
        a = bem_results.a.to_numpy() * f
        R_min = r_inner[0]
        
        p_dynamic_a = 1/2 * rho * v0**2 # dynamic pressure (point A) [N/m2]
        p_stagnation_a = p_static_a + p_dynamic_a # stagnation pressure (point A) [N/m2]
        h_a = p_static_a/rho + v0**2/2 # enthalpy (point A) [J = N*k = kg*m2/s2]
        r_width_b = r_outer- r_inner
        U_b = v0*(1-a) # velocity at disk (Point B/C)[m/s]
        
        r_width_a = U_b*r_width_b / v0
        r_a = np.zeros(len(r_width_a))
        r_a[0] = R_min + r_width_a[0]
        for i in np.arange(1,len(r_width_a)):
            r_a[i] = r_a[i-1] + r_width_a[i]
            
        # Point A
    
        figa.suptitle("Dynamic, Static and Stagnation Pressure (Point A)")
        axa[n,0].plot(p_dynamic_a, r_a/r_outer[-1], linewidth =2, label = f"$\lambda$ = {tsr}", color = color[n], linestyle=linestyle[n])
        axa[n,1].plot(p_static_a, r_a/r_outer[-1], linewidth =2, color = color[n], linestyle=linestyle[n])
        axa[n,2].plot(p_stagnation_a, r_a/r_outer[-1], linewidth =2,  color = color[n], linestyle=linestyle[n])
        axa[n,0].set(ylabel = f"$\lambda$ = {tsr}\n$\mu$ (-)")
        axa[len(tsr_list)-1,0].set(xlabel = r"Dynamic Pressure ($N/m^2$)")
        axa[len(tsr_list)-1,1].set(xlabel = r"Static Pressure ($N/m^2$)")
        axa[len(tsr_list)-1,2].set(xlabel = r"Stagnation Pressure ($N/m^2$)")
        axa[n,0].set_ylim(bottom = 0.175)
        axa[n,0].grid(), axa[n,1].grid(), axa[n,2].grid()
    
    
    
        # Plot Point B
    
        p_static_b = (h_a - U_b**2/2)*rho
        p_dynamic_b = 1/2 * rho * U_b**2
        p_stagnation_b = p_static_b + p_dynamic_b
        h_b = h_a
        
    
        figb.suptitle("Dynamic, Static and Stagnation Pressure (Point B)")
        axb[n,0].plot(p_dynamic_b, r_b/r_outer[-1], linewidth =2, label = f"$\lambda$ = {tsr}", color = color[n], linestyle=linestyle[n])
        axb[n,1].plot(p_static_b, r_b/r_outer[-1], linewidth =2, color = color[n], linestyle=linestyle[n])
        axb[n,2].plot(p_stagnation_b, r_b/r_outer[-1], linewidth =2,  color = color[n], linestyle=linestyle[n])
        axb[n,0].set(ylabel = f"$\lambda$ = {tsr}\n$\mu$ (-)")
        axb[len(tsr_list)-1,0].set(xlabel = r"Dynamic Pressure ($N/m^2$)")
        axb[len(tsr_list)-1,1].set(xlabel = r"Static Pressure ($N/m^2$)")
        axb[len(tsr_list)-1,2].set(xlabel = r"Stagnation Pressure ($N/m^2$)")
        axb[n,0].set_ylim(bottom = 0.175)
        axb[n,0].grid(), axb[n,1].grid(), axb[n,2].grid()
        axb[n,1].ticklabel_format(useOffset=False)
        axb[n,1].xaxis.set_ticks(np.linspace(np.round(min(p_static_b)), np.round(max(p_static_b)), 4))
    
    
        # Plot Point C
        h_c = h_b - f_n/rho
        p_static_c = (h_c - U_b**2/2)*rho
        U_c = U_b
        p_dynamic_c = 1/2 * rho * U_c**2
        p_stagnation_c = p_static_c + p_dynamic_c
    
    
        figc.suptitle("Dynamic, Static and Stagnation Pressure (Point C)")
        axc[n,0].plot(p_dynamic_c, r_b/r_outer[-1], linewidth =2, label = f"$\lambda$ = {tsr}", color = color[n], linestyle=linestyle[n])
        axc[n,1].plot(p_static_c, r_b/r_outer[-1], linewidth =2, color = color[n], linestyle=linestyle[n])
        axc[n,2].plot(p_stagnation_c, r_b/r_outer[-1], linewidth =2,  color = color[n], linestyle=linestyle[n])
        axc[n,0].set(ylabel = f"$\lambda$ = {tsr}\n$\mu$ (-)")
        axc[len(tsr_list)-1,0].set(xlabel = r"Dynamic Pressure ($N/m^2$)")
        axc[len(tsr_list)-1,1].set(xlabel = r"Static Pressure ($N/m^2$)")
        axc[len(tsr_list)-1,2].set(xlabel = r"Stagnation Pressure ($N/m^2$)")
        axc[n,0].grid(), axc[n,1].grid(), axc[n,2].grid()
        axc[n,0].set_ylim(bottom = 0.175)
        axc[n,1].ticklabel_format(useOffset=False)
        axc[n,1].xaxis.set_ticks(np.linspace(np.round(min(p_static_c)), np.round(max(p_static_c)), 5))
        axc[n,2].ticklabel_format(useOffset=False)
        axc[n,2].xaxis.set_ticks(np.linspace(np.round(min(p_stagnation_c)), np.round(max(p_stagnation_c)), 5))
    
    
        # Plot Point D
        U_d = v0*(1-2*a) # velocity in far field [m/s]
        p_dynamic_d = 1/2 * rho * U_d**2 # dynamic pressure (point A) [N/m2]
        p_static_d = p_static_a
        p_stagnation_d = p_static_a + p_dynamic_d
        h_d = p_static_a/rho + U_d**2/2
        
        r_width_d = U_b*r_width_b / U_d
        r_d = np.zeros(len(r_width_a))
        r_d[0] = R_min + r_width_d[0]
        for i in np.arange(1,len(r_width_d)):
            r_d[i] = r_d[i-1] + r_width_d[i]
            
        figd.suptitle("Dynamic, Static and Stagnation Pressure (Point D)")
        axd[n,0].plot(p_dynamic_d, r_d/r_outer[-1], linewidth =2, label = f"$\lambda$ = {tsr}", color = color[n], linestyle=linestyle[n])
        axd[n,1].plot(p_static_d, r_d/r_outer[-1], linewidth =2, color = color[n], linestyle=linestyle[n])
        axd[n,2].plot(p_stagnation_d, r_d/r_outer[-1], linewidth =2,  color = color[n], linestyle=linestyle[n])
        axd[n,0].set(ylabel = f"$\lambda$ = {tsr}\n$\mu$ (-)")
        axd[n,0].set_ylim(bottom = 0.175)
        axd[len(tsr_list)-1,0].set(xlabel = r"Dynamic Pressure ($N/m^2$)")
        axd[len(tsr_list)-1,1].set(xlabel = r"Static Pressure ($N/m^2$)")
        axd[len(tsr_list)-1,2].set(xlabel = r"Stagnation Pressure ($N/m^2$)")
        axd[n,0].grid(), axd[n,1].grid(), axd[n,2].grid()
        
        axd[n,2].ticklabel_format(useOffset=False)
        axd[n,2].xaxis.set_ticks(np.linspace(np.round(min(p_stagnation_d)), np.round(max(p_stagnation_d)), 5))
    
    
    
    
    
    
    figa.tight_layout()
    figa.savefig(path+"/results/point_A.png")
    
    
    figb.tight_layout()
    figb.savefig(path+"/results/point_B.png")
    
    
    figc.tight_layout()
    figc.savefig(path+"/results/point_C.png")
    
    
    figd.tight_layout()
    figd.savefig(path+"/results/point_D.png")