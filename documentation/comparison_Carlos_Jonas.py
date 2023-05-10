import numpy as np

def vortex_induction_factors_Carlos(vortex_start: np.ndarray,
                                    vortex_end: np.ndarray,
                                    induction_point: np.ndarray) -> np.ndarray:
    vec_R_1 = induction_point - vortex_start
    vec_R_2 = induction_point - vortex_end
    R_1 = np.linalg.norm(vec_R_1)
    R_2 = np.linalg.norm(vec_R_2)
    R_1_2 = np.cross(vec_R_1, vec_R_2)
    R_1_2_sqr = np.dot(R_1_2, R_1_2)
    vec_vortex = vortex_end - vortex_start
    R_0_1 = np.dot(vec_vortex, vec_R_1)
    R_0_2 = np.dot(vec_vortex, vec_R_2)
    K = 1/(4*np.pi*R_1_2_sqr)*(R_0_1/R_1-R_0_2/R_2)
    return K*R_1_2

def vortex_induction_factors_Jonas(vortex_start: np.ndarray,
                                   vortex_end: np.ndarray,
                                   induction_point: np.ndarray) -> np.ndarray:
    r_s = induction_point-vortex_start
    r_e = induction_point-vortex_end
    r_v = vortex_end-vortex_start

    l_s = np.linalg.norm(r_s)
    l_e = np.linalg.norm(r_e)
    l_v = np.linalg.norm(r_v)

    h = np.linalg.norm(np.cross(r_v, r_s))/l_v
    e_i = np.cross(r_v, r_s)/(h*l_v)
    return e_i/(4*np.pi*h*l_v)*(np.dot(r_v, (r_s/l_s-r_e/l_e)))

n_vortex_elements = 100
control_point = np.random.random(3)
all_same = True
print_factors = True
for n in range(n_vortex_elements):
    vortex_start = np.random.random(3)
    vortex_end = np.random.random(3)
    facs_Carlos = vortex_induction_factors_Carlos(vortex_start, vortex_end, control_point)
    facs_Jonas = vortex_induction_factors_Jonas(vortex_start, vortex_end, control_point)
    if np.linalg.norm(facs_Carlos-facs_Jonas) > 1e-13:
        print(f"Induction factors are different for vortex from {vortex_start} to {vortex_end}", "\n",
              f"Carlos: {facs_Carlos}", "\n",
              f"Jonas: {facs_Jonas}")
        all_same = False
    if print_factors:
        print(f"Carlos: {facs_Carlos}", "\n", f"Jonas: {facs_Jonas}", "\n")
if all_same:
    print("All induction factors are the same")