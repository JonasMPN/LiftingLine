
    def optimize_TUD(self,
                  wind_speed: float,
                  tip_speed_ratio: float,
                  pitch: float or np.ndarray,
                  start_radius: float = None,
                  resolution: int = 200,
                  max_convergence_error: float=1e-5,
                  max_iterations: int=200,
                  tip_loss_correction: bool=True,
                  root_loss_correction: bool=True) -> None:
        """
        Optimze the chord and twist per blade section
        Glauert_correction: either 'tud' (TU Delft) or 'dtu' (Denmark's TU). Same for blade_end_correction
        All angles must be in rad.
        :param wind_speed:
        :param tip_speed_ratio:
        :param pitch: IN DEGREE
        :param resolution:
        :return:
        """
        start_radius = start_radius if start_radius is not None else self.root_radius
        # Initialise the result containers
        results = {
            "r_centre": list(),     # radius used for the calculations
            "r_inner": list(),      # inner radius of the blade element
            "r_outer": list(),      # outer radius of the blade element
            "a": list(),            # Axial Induction factor
            "a_prime": list(),      # Tangential induction factor
            "f_n": list(),          # Forces normal to the rotor plane in N/m
            "f_t": list(),          # Forces tangential in the rotor plane in N/m
            "bec": list(),          # blade end correction (depending on 'tip' and 'root')
            "C_T": list(),          # thrust coefficient
            "alpha": list(),        # angle of attack
            "circulation": list(),  # magnitude of the circulation using Kutta-Joukowski
            "v0": list(),           # flow velocity normal to rotor plane
            "tsr": list(),          # tip speed ratio
            "pitch": list(),         # pitch in degree
            "end_correction": list(),# Prandtl tip and root loss factor
            "cl": list(),
            "cd": list()
        }
        # delete data with same wind speed, tip speed ratio and pitch angle.
        try:
            self.df_results = self.df_results.loc[~((self.df_results["tsr"]==tip_speed_ratio) &
                                                    (self.df_results["v0"]==wind_speed) &
                                                    (self.df_results["pitch"]==pitch))]
        except KeyError:
            pass
        pitch = np.deg2rad(pitch)
        # Calculate the rotational speed
        #breakpoint()

        omega = tip_speed_ratio*wind_speed/self.rotor_radius
        # go from middle from now
        #radii = np.linspace(0.55*(self.rotor_radius - start_radius), self.rotor_radius, resolution)
        radii = np.linspace(start_radius, self.rotor_radius, resolution)
        radii_left = radii[:int((resolution/2)+1)] # have one element overlap
        radii_right = radii[int(resolution/2):]

        # initialize arrays to store the optimum values
        chord_list = np.zeros(len(radii))
        twist_list = np.zeros(len(radii))
        alpha_list = np.zeros(len(radii))
        # Loop along the span of the blade (blade sections)
        print(f"Doing BEM for v0={wind_speed}, tsr={tip_speed_ratio}, pitch={pitch}")

        # loop through outer part

        #for r_inside, r_outside in zip(radii[:-1], radii[1:]):      # Take the left and right radius of every element
        iter = 0
        chord = 3.24
        twist = 0.055

        # Loop for the outer section
        breakpoint()
        for r_inside, r_outside in zip(radii_right[:-1], radii_right[1:]):      # Take the left and right radius of every element
            #breakpoint()
            r_centre = (r_inside+r_outside)/2                       # representative radius in the middle of the section
            elem_length = r_outside-r_inside                        # length of elemen
            # Get/Set values from the local section



            def get_force_from_bem(chord, twist): #chord_and_twist):
                """
                Function to return the tangential force for a given chord and twist
                """

                #chord = chord_and_twist[0]
                #twist = chord_and_twist[1]
                try:
                    area_annulus = np.pi*(r_outside**2-r_inside**2)
                    a, a_new, a_prime, converged = 1/3, 0, 0, False
                    for i in range(max_iterations):
                        # get inflow angle
                        phi , speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
                        # get combined lift and drag coefficient projected into the normal and tangential direction
                        _, _, _, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch,
                                                                     tip_seed_ratio=tip_speed_ratio, university="tud")
                        # get the inflow speed for the airfoil
                        inflow_speed = self._inflow_velocity(wind_speed, a, a_prime, omega, r_centre)
                        # get thrust force (in N) of the whole turbine at the current radius
                        thrust = self._aero_force(inflow_speed, chord, c_n)*self.n_blades*elem_length
                        # calculate thrust coefficient that results from the blade element
                        C_T = thrust/(1/2*self.air_density*wind_speed**2*area_annulus)
                        # get Glauert corrected axial induction factor
                        a_new = self._a(C_T=C_T)
                        # get the combined (tip and root) correction factor
                        blade_end_correction = self._blade_end_correction(which="tud", tip=tip_loss_correction,
                                                                          root=root_loss_correction, radius=r_centre,
                                                                          tip_seed_ratio=tip_speed_ratio, a=a_new)
                        # correct the Glauert corrected axial induction factor with the blade end losses
                        a_new /= blade_end_correction
                        # update the axial induction factor for the next iteration
                        a = 0.75*a+0.25*a_new
                        # get the tangential force (in N/m) of the whole turbine at the current radius
                        f_tangential = self._aero_force(inflow_speed, chord, c_t)*self.n_blades
                        # get the tangential induction factor that corresponds to this force AND correct it for tip losses
                        a_prime = self._a_prime(f_tangential, r_centre, wind_speed, a, tip_speed_ratio)/blade_end_correction
                        # check if the axial induction factor has converged. If it has, the tangential induction factor has too
                        if np.abs(a-a_new) < max_convergence_error:
                            converged = True
                            break
                    phi, speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
                    alpha, c_l, c_d, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch, radius=r_centre,
                                                                       tip_seed_ratio=tip_speed_ratio, university="tud")
                    return f_tangential, alpha , a ,a_prime, converged, blade_end_correction  # output the tangential force
                except:
                    return np.nan, np.nan, 0, 0 , False
            # Brute force through it !
            #chord_estimate = 4.8
            #twist_estimate = 0.15
            #chord_range = np.linspace(chord_estimate*0.5, chord_estimate*1.5,40)
            #twist_range = np.linspace(twist_estimate * 0.5, twist_estimate *1.5, 40) #np.deg2rad(np.linspace(7.4,11.4,100))
            c_l_opt = 1.1
            #chord_estimate = self._calc_optimum_chord(r_centre,self.rotor_radius,self.n_blades,c_l_opt,tip_speed_ratio)
            chord_range = np.arange((chord)*0.9, (chord)*1.15,0.1) # 5cm steps should be alright resolution
            twist_range = np.arange(twist* 0.9, twist*1.15, np.deg2rad(0.75)) #np.deg2rad(np.linspace(7.4,11.4,100)) # 1 degree steps
            #chord_range = [2,3,4,5]
            #twist_range = [0.13,0.14,0.15, 0.16]
            ft_array =np.empty((len(chord_range), len(twist_range))) # initialize chord x twist array
            alpha_array =np.empty((len(chord_range), len(twist_range))) # initialize chord x twist array
            for j,twist in enumerate(twist_range):
                for i,chord in enumerate(chord_range):
                    ft_array[i,j], alpha_array[i,j] , a , a_prime, converged, blade_end_correction= get_force_from_bem(chord,twist)


            # Now we have these optimums. Get the values of chord and twist
            max_indices = np.unravel_index(np.argmax(ft_array, axis=None), ft_array.shape)
            chord_list[iter+int(resolution/2)]  = chord_range[max_indices[0]]
            twist_list[iter+int(resolution/2)]  = twist_range[max_indices[1]]
            alpha_list[iter+int(resolution/2)]  = alpha_array[max_indices]

            # reset to optimum values
            chord  = chord_range[max_indices[0]] 
            twist  = twist_range[max_indices[1]]
            aoa = alpha_array[max_indices]
            print(f"Estimates:\n Chord: {chord} \n Twist: {twist}")
            print(f"Op. value:\n r: {r_centre} \n AOA: {aoa}")
            #import matplotlib.pyplot as plt
            #plt.contourf(np.rad2deg(twist_range), chord_range,ft_array,20)
            #plt.show()

            #initial_guess = 5
            #bounds = (0,1)
            #optimum = minimize(get_force_from_bem,initial_guess, method="TNC")
            #breakpoint()
            # notify user if loop did not converge, but was stopped by the maximum number of iterations
            # if not converged:
            #     print(f"BEM did not converge for the blade element between {r_inside}m and {r_outside}m. Current "
            #           f"change after {max_iterations}: {np.abs(a-a_new)}.")

            # Now that we have the converged axial induction factor, we can get the rest of the values
            phi, speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
            alpha, c_l, c_d, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch, radius=r_centre,
                                                               tip_seed_ratio=tip_speed_ratio, university="tud")
            inflow_speed = self._inflow_velocity(wind_speed, a, a_prime, omega, r_centre)

            # Assemble the result output structure
            results["r_centre"].append(r_centre)
            results["r_inner"].append(r_inside)
            results["r_outer"].append(r_outside)
            results["a"].append(a)
            results["a_prime"].append(a_prime)
            results["f_n"].append(self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=c_n))
            results["f_t"].append(self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=c_t))
            results["bec"].append(self._blade_end_correction(which="tud", tip=tip_loss_correction,
                                                             root=root_loss_correction, radius=r_centre,
                                                             tip_seed_ratio=tip_speed_ratio, a=a))
            results["C_T"].append(self._C_T(a))
            results["alpha"].append(alpha)
            results["cl"].append(c_l)
            results["cd"].append(c_d)
            results["circulation"].append(1/2*inflow_speed*c_l*chord)
            results["v0"].append(wind_speed)
            results["tsr"].append(tip_speed_ratio)
            results["pitch"].append(np.rad2deg(pitch))
            results["end_correction"].append(blade_end_correction)
            iter +=1


        chord = 3.24
        twist = 0.055
        breakpoint()

        # Loop for the inner section
        iter = 0
        for r_inside, r_outside in zip(radii_left[:-1][::-1], radii_left[1:][::-1]):      # Take the left and right radius of every element
            #breakpoint()
            r_centre = (r_inside+r_outside)/2                       # representative radius in the middle of the section
            elem_length = r_outside-r_inside                        # length of elemen
            # Get/Set values from the local section



            def get_force_from_bem(chord, twist): #chord_and_twist):
                """
                Function to return the tangential force for a given chord and twist 
                """

                #chord = chord_and_twist[0]
                #twist = chord_and_twist[1]
                try:
                    area_annulus = np.pi*(r_outside**2-r_inside**2)
                    a, a_new, a_prime, converged = 1/3, 0, 0, False
                    for i in range(max_iterations):
                        # get inflow angle
                        phi, speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
                        # get combined lift and drag coefficient projected into the normal and tangential direction
                        _, _, _, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch,
                                                                     tip_seed_ratio=tip_speed_ratio, university="tud")
                        # get the inflow speed for the airfoil
                        inflow_speed = self._inflow_velocity(wind_speed, a, a_prime, omega, r_centre)
                        # get thrust force (in N) of the whole turbine at the current radius
                        thrust = self._aero_force(inflow_speed, chord, c_n)*self.n_blades*elem_length
                        # calculate thrust coefficient that results from the blade element
                        C_T = thrust/(1/2*self.air_density*wind_speed**2*area_annulus)
                        # get Glauert corrected axial induction factor
                        a_new = self._a(C_T=C_T)
                        # get the combined (tip and root) correction factor
                        blade_end_correction = self._blade_end_correction(which="tud", tip=tip_loss_correction,
                                                                          root=root_loss_correction, radius=r_centre,
                                                                          tip_seed_ratio=tip_speed_ratio, a=a_new)
                        # correct the Glauert corrected axial induction factor with the blade end losses
                        a_new /= blade_end_correction
                        # update the axial induction factor for the next iteration
                        a = 0.75*a+0.25*a_new
                        # get the tangential force (in N/m) of the whole turbine at the current radius
                        f_tangential = self._aero_force(inflow_speed, chord, c_t)*self.n_blades
                        # get the tangential induction factor that corresponds to this force AND correct it for tip losses
                        a_prime = self._a_prime(f_tangential, r_centre, wind_speed, a, tip_speed_ratio)/blade_end_correction
                        # check if the axial induction factor has converged. If it has, the tangential induction factor has too
                        if np.abs(a-a_new) < max_convergence_error:
                            converged = True
                            break
                    phi, speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
                    alpha, c_l, c_d, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch, radius=r_centre,
                                                                       tip_seed_ratio=tip_speed_ratio, university="tud")
                    return f_tangential, alpha , a ,a_prime, converged, blade_end_correction  # output the tangential force
                except:
                    return np.nan, np.nan, 0, 0 , True, 0
            # Brute force through it ! 
            #chord_estimate = 4.8
            #twist_estimate = 0.15
            #chord_range = np.linspace(chord_estimate*0.5, chord_estimate*1.5,40)
            #twist_range = np.linspace(twist_estimate * 0.5, twist_estimate *1.5, 40) #np.deg2rad(np.linspace(7.4,11.4,100))
            c_l_opt = 1.1
            #chord_estimate = self._calc_optimum_chord(r_centre,self.rotor_radius,self.n_blades,c_l_opt,tip_speed_ratio)
            chord_range = np.arange((chord)*0.9, (chord)*1.1,0.10) # 5cm steps should be alright resolution
            twist_range = np.arange(twist* 0.9, twist*1.1, np.deg2rad(0.75)) #np.deg2rad(np.linspace(7.4,11.4,100)) # 1 degree steps
            #chord_range = [2,3,4,5]
            #twist_range = [0.13,0.14,0.15, 0.16]
            ft_array =np.empty((len(chord_range), len(twist_range))) # initialize chord x twist array
            alpha_array =np.empty((len(chord_range), len(twist_range))) # initialize chord x twist array
            for j,twist in enumerate(twist_range):
                for i,chord in enumerate(chord_range):
                    try:
                        ft_array[i,j], alpha_array[i,j] , a , a_prime, converged, blade_end_correction= get_force_from_bem(chord,twist)
                    except:
                        ft_array[i,j], alpha_array[i,j] = 0,0
               

            # Now we have these optimums. Get the values of chord and twist
            #breakpoint()
            max_indices = np.unravel_index(np.argmax(ft_array, axis=None), ft_array.shape)
            chord_list[iter+int(resolution/2)]  = chord_range[max_indices[0]]
            twist_list[iter+int(resolution/2)]  = twist_range[max_indices[1]]
            alpha_list[iter+int(resolution/2)]  = alpha_array[max_indices]

            # reset to optimum values
            chord  = chord_range[max_indices[0]] 
            twist  = twist_range[max_indices[1]]
            aoa = alpha_array[max_indices]
            print(f"Estimates:\n Chord: {chord} \n Twist: {twist}")
            print(f"Op. value:\n r: {r_centre} \n AOA: {aoa}")
            
            #import matplotlib.pyplot as plt
            #plt.contourf(np.rad2deg(twist_range), chord_range,ft_array,20)
            #plt.xlabel("Twist [deg]")
            #plt.ylabel("Chord [m]")
            #plt.show()
            
            #initial_guess = 5
            #bounds = (0,1)
            #optimum = minimize(get_force_from_bem,initial_guess, method="TNC")
            #breakpoint()
            # notify user if loop did not converge, but was stopped by the maximum number of iterations
            if not converged:
                print(f"BEM did not converge for the blade element between {r_inside}m and {r_outside}m. Current "
                      f"change after {max_iterations}: {np.abs(a-a_new)}.")

            # Now that we have the converged axial induction factor, we can get the rest of the values
            phi, speed = self._flow(a=a, a_prime=a_prime, wind_speed=wind_speed, rotational_speed=omega, radius=r_centre)
            try:
                alpha, c_l, c_d, c_n, c_t = self._phi_to_aero_values(phi=phi, twist=twist, pitch=pitch, radius=r_centre,
                                                                   tip_seed_ratio=tip_speed_ratio, university="tud")
            except:
                print("Schade, das war wohl nix!")
            inflow_speed = self._inflow_velocity(wind_speed, a, a_prime, omega, r_centre)

            # Assemble the result output structure
            results["r_centre"].append(r_centre)
            results["r_inner"].append(r_inside)
            results["r_outer"].append(r_outside)
            results["a"].append(a)
            results["a_prime"].append(a_prime)
            results["f_n"].append(self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=c_n))
            results["f_t"].append(self._aero_force(inflow_velocity=inflow_speed, chord=chord, force_coefficient=c_t))
            results["bec"].append(self._blade_end_correction(which="tud", tip=tip_loss_correction,
                                                             root=root_loss_correction, radius=r_centre,
                                                             tip_seed_ratio=tip_speed_ratio, a=a))
            results["C_T"].append(self._C_T(a))
            results["alpha"].append(alpha)
            results["cl"].append(c_l)
            results["cd"].append(c_d)
            results["circulation"].append(1/2*inflow_speed*c_l*chord)
            results["v0"].append(wind_speed)
            results["tsr"].append(tip_speed_ratio)
            results["pitch"].append(np.rad2deg(pitch))
            results["end_correction"].append(blade_end_correction)
            iter -=1
        # plot intermed. results
        fig, axs = plt.subplots(3,1)
        axs[0].plot(alpha_list)
        axs[1].plot(twist_list)
        axs[2].plot(chord_list)
        axs[0].set_ylabel("AOA")
        axs[1].set_ylabel("Twist")
        axs[2].set_ylabel("Chord")
        plt.show()

        breakpoint()
        self.current_results = pd.DataFrame(results)
        self.df_results = pd.concat([self.df_results, pd.DataFrame(results)])
        self.df_results.to_csv(self.root+"/BEM_results.dat", index=False)
        return None
