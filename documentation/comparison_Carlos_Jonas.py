import numpy as np

def vortex_induction_factors_Carlos(vortex_start: np.ndarray,
                                    vortex_end: np.ndarray,
                                    induction_point: np.ndarray) -> np.ndarray:
    X1, Y1, Z1 = vortex_start[0], vortex_start[1], vortex_start[2]
    X2, Y2, Z2 = vortex_end[0], vortex_end[1], vortex_end[2]
    XP, YP, ZP = induction_point[0], induction_point[1], induction_point[2]
    R1 = np.sqrt((XP-X1)**2+(YP-Y1)**2+(ZP-Z1)**2)
    R2 = np.sqrt((XP-X2)**2+(YP-Y2)**2+(ZP-Z2)**2)
    R12_X = (YP-Y1)*(ZP-Z2)-(ZP-Z1)*(YP-Y2)
    R12_Y = -(XP-X1)*(ZP-Z2)+(ZP-Z1)*(XP-X2)
    R12_Z = (XP-X1)*(YP-Y2)-(YP-Y1)*(XP-X2)
    R12_sqr = R12_X**2+R12_Y**2+R12_Z**2
    R0R1 = (X2-X1)*(XP-X1)+(Y2-Y1)*(YP-Y1)+(Z2-Z1)*(ZP-Z1)
    R0R2 = (X2-X1)*(XP-X2)+(Y2-Y1)*(YP-Y2)+(Z2-Z1)*(ZP-Z2)
    K = 1/(4*np.pi*R12_sqr)*(R0R1/R1-R0R2/R2)
    return np.asarray([K*R12_X, K*R12_Y, K*R12_Z])

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