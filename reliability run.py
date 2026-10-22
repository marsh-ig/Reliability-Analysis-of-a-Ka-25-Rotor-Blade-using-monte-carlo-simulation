from ansys.mapdl.core import launch_mapdl
import numpy as np
import pandas as pd

# ==========================================
# 1. XFOIL NACA 23012 Polar Data
# ==========================================
polar_alpha = np.array([
    -10.25, -10.0, -9.75, -9.5, -9.25, -9.0, -8.75, -8.5, -8.25, -8.0, -7.75, -7.5, -7.25, -6.5, -6.25, -6.0, -5.75, -5.5, -5.25, -5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25
])
polar_CL = np.array([
    -0.5351, -0.5476, -0.6133, -0.5441, -0.5469, -0.6158, -0.5617, -0.5836, -0.6066, -0.6091, -0.5924, -0.5907, -0.5874, -0.6046, -0.5866, -0.5711, -0.5542, -0.5360, -0.5157, -0.4952, -0.4742, -0.4524, -0.4301, -0.4080, -0.3860, -0.3645, -0.3438, -0.3237, -0.3044, -0.2845, -0.2685, -0.2510, -0.1923, -0.0909, 0.0421, 0.1135, 0.1774, 0.2269, 0.2638, 0.2908, 0.3125, 0.3327, 0.3528, 0.3734, 0.3943, 0.4153, 0.4363, 0.4574, 0.4786, 0.4999, 0.5215, 0.5432, 0.5653, 0.5876, 0.6100, 0.6325, 0.6552, 0.6781, 0.7010, 0.7237, 0.7476, 0.7694, 0.7931, 0.8149, 0.8386, 0.8601, 0.8843, 0.9047, 0.9296, 0.9486, 0.9727, 0.9916, 1.0146, 1.0331, 1.0549, 1.0734, 1.0935, 1.1118, 1.1301, 1.1478, 1.1648, 1.1805, 1.1975, 1.2093, 1.2272, 1.2335, 1.2514, 1.2516, 1.2629, 1.2637, 1.2513, 1.2742, 1.2490, 1.2259, 1.2048, 1.0684, 1.0478
])
polar_CD = np.array([
    0.10845, 0.10393, 0.09584, 0.09556, 0.09161, 0.08385, 0.08209, 0.07723, 0.07210, 0.06945, 0.06694, 0.06417, 0.06172, 0.04013, 0.03707, 0.03447, 0.03210, 0.03014, 0.02842, 0.02687, 0.02563, 0.02442, 0.02338, 0.02237, 0.02120, 0.02046, 0.01956, 0.01895, 0.01825, 0.01761, 0.01498, 0.01491, 0.01576, 0.01666, 0.01689, 0.01648, 0.01597, 0.01547, 0.01502, 0.01466, 0.01440, 0.01421, 0.01410, 0.01407, 0.01411, 0.01423, 0.01441, 0.01465, 0.01495, 0.01530, 0.01569, 0.01608, 0.01650, 0.01695, 0.01740, 0.01787, 0.01837, 0.01890, 0.01941, 0.01998, 0.02060, 0.02115, 0.02184, 0.02244, 0.02318, 0.02385, 0.02465, 0.02538, 0.02619, 0.02704, 0.02778, 0.02883, 0.02952, 0.03068, 0.03138, 0.03256, 0.03329, 0.03443, 0.03520, 0.03628, 0.03699, 0.03808, 0.03863, 0.03990, 0.04028, 0.04189, 0.04218, 0.04416, 0.04483, 0.04673, 0.04892, 0.04924, 0.05267, 0.05661, 0.06092, 0.08320, 0.09022
])
polar_CM = np.array([
    -0.0041, -0.0085, -0.0235, -0.0101, -0.0122, -0.0239, -0.0172, -0.0205, -0.0224, -0.0219, -0.0196, -0.0182, -0.0165, -0.0135, -0.0117, -0.0095, -0.0073, -0.0049, -0.0032, -0.0020, -0.0006, 0.0007, 0.0020, 0.0032, 0.0041, 0.0053, 0.0064, 0.0075, 0.0086, 0.0095, 0.0110, 0.0160, 0.0130, 0.0010, -0.0177, -0.0269, -0.0344, -0.0388, -0.0404, -0.0398, -0.0381, -0.0361, -0.0341, -0.0322, -0.0305, -0.0289, -0.0274, -0.0259, -0.0245, -0.0233, -0.0220, -0.0209, -0.0198, -0.0188, -0.0178, -0.0169, -0.0160, -0.0151, -0.0142, -0.0134, -0.0127, -0.0116, -0.0109, -0.0099, -0.0093, -0.0083, -0.0077, -0.0066, -0.0062, -0.0049, -0.0044, -0.0032, -0.0025, -0.0014, -0.0006, 0.0005, 0.0015, 0.0026, 0.0038, 0.0049, 0.0062, 0.0075, 0.0087, 0.0104, 0.0115, 0.0136, 0.0147, 0.0172, 0.0188, 0.0210, 0.0243, 0.0249, 0.0278, 0.0290, 0.0289, 0.0169, 0.0133
])
nominal_liftslope = (
    np.interp(5.0, polar_alpha, polar_CL) - np.interp(1.0, polar_alpha, polar_CL)
) / np.radians(4.0)

ka25_data = {
    'radius': 7.87,
    'omega': 24.82,
    'chord': 0.370,
    'hinge_offset': 0.785,
    'mass_density': 65.3 / (7.87 - 0.785),  # Linear mass density [kg/m]
    'number_of_blades': 3,
    'air_density': 1.225,
    'root_pitch': np.radians(10.0),          # Collective [rad]
    'tip_pitch': np.radians(-0.75),
    'flap_pitch_coupling': np.radians(0.0),
    'cyclic_offset': 0.0,
    'hub_x': 0.0, 'hub_y': 0.0, 'hub_z': 0.0
}
Area = 1.049e-3
Iyy  = 1.88e-6
Iyz  = 1.002e-8
Izz  = 2.15e-7
J    = 2.10e-6
TKZ  = 0.370     # Section extent in local z (chord)
TKY  = 0.0444    # Section extent in local y (thickness)

denom = Iyy * Izz - Iyz**2
c_y = TKY / 2.0
c_z = TKZ / 2.0

def get_blade_velocity_2k(main_data, body_vel, body_rot, flap, psi, x_k):
    MR_offset = main_data['hinge_offset']
    MR_omega = main_data['omega']

    beta, dbeta, _ = flap[0], flap[1], flap[2]

    V_cg = np.array(body_vel)
    Omega_cg = np.array(body_rot)

    # FIX: Separate hub-to-hinge vector from hinge-to-section vector
    R_hinge_1 = np.array([MR_offset, 0.0, 0.0])
    R_p_2k = np.array([x_k - MR_offset, 0.0, 0.0])

    T_1_H = np.array([
        [ np.cos(psi), np.sin(psi), 0.0],
        [-np.sin(psi), np.cos(psi), 0.0],
        [ 0.0,         0.0,         1.0]
    ])

    T_2k_theta = np.array([
        [ np.cos(beta), 0.0, np.sin(beta)],
        [ 0.0,          1.0, 0.0         ],
        [-np.sin(beta), 0.0, np.cos(beta)]
    ])

    Rpm = np.array([0.0, 0.0, MR_omega])
    beta_dot_vec = np.array([0.0, -dbeta, 0.0])

    Omega_hub_1 = (T_1_H @ Omega_cg) + Rpm
    V_hub_1 = T_1_H @ V_cg
    V_hinge_1 = V_hub_1 + np.cross(Omega_hub_1, R_hinge_1)

    Omega_hinge_2k = (T_2k_theta @ Omega_hub_1) + beta_dot_vec
    V_hinge_2k = T_2k_theta @ V_hinge_1

    V_p_2k = V_hinge_2k + np.cross(Omega_hinge_2k, R_p_2k)
    return V_p_2k


def get_sectional_aero_loads(main_data, body_vel, body_rot, flap, psi, x_k):
    MR_radius = main_data['radius']
    MR_omega = main_data['omega']
    MR_chord = main_data['chord']
    NoB = main_data['number_of_blades']
    Air_density = main_data['air_density']
    MR_flappitchcoupling = main_data['flap_pitch_coupling']

    beta, dbeta, ddbeta = flap[0], flap[1], flap[2]
    root_cutout = 1.966

    if x_k < root_cutout:
        return np.zeros(6)

    T_2k_theta = np.array([
        [ np.cos(beta), 0.0, np.sin(beta)],
        [ 0.0,          1.0, 0.0         ],
        [-np.sin(beta), 0.0, np.cos(beta)]
    ])

    # 1. Piecewise Twist Logic (FIX: r_twist_end = 0.7 * MR_radius)
    r_twist_start = 0.3 * MR_radius
    r_twist_end = 1 * MR_radius
    twist_total_rad = np.radians(-8.75)

    if x_k <= r_twist_start:
        theta_local = main_data['root_pitch']
    elif r_twist_start < x_k <= r_twist_end:
        twist_fraction = (x_k - r_twist_start) / (r_twist_end - r_twist_start)
        theta_local = main_data['root_pitch'] + (twist_total_rad * twist_fraction)
    else:
        theta_local = main_data['root_pitch'] + twist_total_rad

    # FIX: Include cyclic_offset and flap-pitch coupling consistently in theta_tot
    theta_tot = (
        theta_local
        + main_data['cyclic_offset']
        - np.tan(MR_flappitchcoupling) * beta
    )

    # 2. BEMT Inflow with Prandtl Tip Loss
    sigma = (NoB * MR_chord) / (np.pi * MR_radius)
    r_bar = x_k / MR_radius

    bemt_inner = 1.0 + (32.0 / (sigma * nominal_liftslope)) * (theta_tot + np.radians(1.35)) * r_bar
    bemt_inner = max(bemt_inner, 0.0)
    lambda_i = max((sigma * nominal_liftslope / 16.0) * (np.sqrt(bemt_inner) - 1.0), 0.0)

    V_P_2k = get_blade_velocity_2k(main_data, body_vel, body_rot, flap, psi, x_k)

    inflow_temp = np.array([0.0, 0.0, MR_omega * MR_radius * lambda_i])
    lam_temp1 = T_2k_theta @ inflow_temp
    U = V_P_2k + lam_temp1

    # 3. Angle of Attack & Prandtl Tip Loss
    phi = np.arctan2(U[2], U[1]) if U[1] != 0.0 else 0.0
    f_tip = (NoB * (MR_radius - x_k)) / (2.0 * x_k * np.sin(abs(phi) + 1e-6))
    F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0))

    alpha_rad = theta_tot - phi
    alpha_deg = np.degrees(alpha_rad)

    U_res = np.sqrt(U[2]**2 + U[1]**2)

    # 4. Interpolate XFOIL Data
    Cl = np.interp(alpha_deg, polar_alpha, polar_CL)
    Cd = np.interp(alpha_deg, polar_alpha, polar_CD)
    Cm = np.interp(alpha_deg, polar_alpha, polar_CM)

    # 5. Calculate Sectional Forces & Moments
    dyn_pres = 0.5 * Air_density * MR_chord * (U_res**2)
    Lift = F * dyn_pres * Cl
    Drag = dyn_pres * Cd
    Pitching_Moment = F * dyn_pres * MR_chord * Cm

    Aero_F_2k = np.array([
        0.0,
        -(Lift * np.sin(phi) + Drag * np.cos(phi)),
        (Lift * np.cos(phi) - Drag * np.sin(phi))
    ])
    Aero_M_2k = np.array([Pitching_Moment, 0.0, 0.0])

    # Rotate back to hub plane (T_theta_2k = T_2k_theta.T)
    T_theta_2k = T_2k_theta.T
    Aero_F_theta = T_theta_2k @ Aero_F_2k
    Aero_M_theta = T_theta_2k @ Aero_M_2k

    # FIX: Return pure sectional moment Aero_M_theta (not hinge cross-product moment)
    return np.concatenate((Aero_F_theta, Aero_M_theta))

# 1. THE BOX-BEHNKEN RUN MATRIX
# [Velocity (m/s), RPM (rad/s), Pitch (deg)]
doe_matrix = [
    [15.0, 23.50, 10.0], [40.0, 23.50, 10.0],
    [15.0, 26.10, 10.0], [40.0, 26.10, 10.0],
    [15.0, 24.82,  8.0], [40.0, 24.82,  8.0],
    [15.0, 24.82, 12.0], [40.0, 24.82, 12.0],
    [27.5, 23.50,  8.0], [27.5, 26.10,  8.0],
    [27.5, 23.50, 12.0], [27.5, 26.10, 12.0],
    [27.5, 24.82, 10.0]  # Center point
]

# 2. LAUNCH MAPDL ONCE 
mapdl = launch_mapdl()
run_results = []

# 3. THE AUTOMATION LOOP
for run_id, (vel, rpm, pitch_deg) in enumerate(doe_matrix):
    print(f"\n=======================================================")
    print(f" RUN {run_id+1}/13 | Vel: {vel} m/s | RPM: {rpm} | Pitch: {pitch_deg} deg")
    print(f"=======================================================")
    
    # Wipe the ANSYS memory clean for the new run
    mapdl.clear()
    
    # --- UPDATE YOUR DICTIONARY FOR THIS RUN ---
    ka25_data['omega'] = rpm
    ka25_data['root_pitch'] = np.radians(pitch_deg)
    body_vel = [vel, 0.0, 0.0]

    # ==========================================
    # PHASE 1: INITIALIZE PyMAPDL & MESH
    # ==========================================
    
    mapdl.clear()
    mapdl.prep7()
    mapdl.units("SI")

    r_root = ka25_data['hinge_offset']
    r_tip = ka25_data['radius']
    n_elements = 50

    mapdl.mp("EX", 1, 71e9)
    mapdl.mp("PRXY", 1, 0.33)
    mapdl.mp("DENS", 1, ka25_data["mass_density"] / Area)

    mapdl.sectype(1, "BEAM", "ASEC")
    mapdl.secdata(Area, Iyy, Iyz, Izz, 0.0, J, 0.0, 0.0, 0.0, 0.0, TKZ, TKY)

    mapdl.k(1, r_root, 0.0, 0.0)
    mapdl.k(2, r_tip, 0.0, 0.0)
    mapdl.k(3, r_root, 1.0, 0.0)

    mapdl.l(1, 2)
    mapdl.et(1, 188)
    mapdl.latt(mat=1, secnum=1, kb=3)
    mapdl.lesize(1, ndiv=n_elements)
    mapdl.lmesh(1)

    # Extract & cache ordered beam nodes
    dl = (r_tip - r_root) / n_elements
    x_nodes = np.linspace(r_root, r_tip, n_elements + 1)
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    all_node_ids = mapdl.mesh.nnum
    all_node_coords = mapdl.mesh.nodes
    beam_mask = np.abs(all_node_coords[:, 1]) < 1e-6
    beam_node_ids = all_node_ids[beam_mask]
    beam_node_coords = all_node_coords[beam_mask]

    sort_indices = np.argsort(beam_node_coords[:, 0])
    tracked_nodes = beam_node_ids[sort_indices]
    tracked_x = beam_node_coords[sort_indices, 0]
    root_node = int(tracked_nodes[0])

    # Boundary Conditions: Constrain root node; allow blade lead-lag & torsion or lock at root
    mapdl.d(root_node, "ALL", 0.0)

    print(f"Phase 1 Complete: Meshed {n_elements} BEAM188 elements from x={r_root} to x={r_tip}.")

    # ==========================================
    # PHASE 2 & 3: AEROELASTIC FSI SOLVER
    # ==========================================
    print("\nStarting Fluid-Structure Interaction (FSI) Loop...")

    mapdl.run("/SOLU")
    mapdl.antype("TRAN")
    mapdl.trnopt("FULL")
    mapdl.nlgeom("ON")
    mapdl.outres("ALL", "ALL")

    # Step 1: Spin Up (Hinges Locked)
    mapdl.time(1.0)
    mapdl.omega(0.0, ka25_data['omega'], 0.0)
    mapdl.alphad(0.5)
    mapdl.betad(1e-4)
    mapdl.kbc(0)
    mapdl.nsubst(10, 20, 5)
    mapdl.solve()

    # Step 2: Unlock Flap Hinge (FIX: Use ddele on root_node & remove artificial mass damping)
    mapdl.time(1.1)
    mapdl.ddele(root_node, "ROTZ")
    mapdl.alphad(0.0)   # Turn off artificial mass damping; aerodynamic damping handles flap rate
    mapdl.betad(1e-4)   # Small structural stiffness damping for high-frequency FE modes
    mapdl.kbc(1)
    mapdl.nsubst(5, 10, 5)
    mapdl.solve()

    mapdl.mute = True

    num_nodes = len(tracked_nodes)
    # Initialize arrays in MAPDL
    mapdl.run(f"*DIM, NODE_LIST, ARRAY, {num_nodes}")
    mapdl.parameters['NODE_LIST'] = tracked_nodes

    mapdl.run(f"*DIM, UY_ALL, ARRAY, {num_nodes}")
    mapdl.run(f"*DIM, VY_ALL, ARRAY, {num_nodes}")
    mapdl.run(f"*DIM, AY_ALL, ARRAY, {num_nodes}")

    # --- TIME-MARCHING KINEMATICS (COMMENT #1: Revolutions as factor of 360) ---
    dpsi_deg = 5.0
    dpsi_rad = np.radians(dpsi_deg)
    dt = dpsi_rad / ka25_data['omega']

    n_revs = 3
    steps_per_rev = int(round(360.0 / dpsi_deg))
    azimuth_steps = np.radians(np.arange(0.0, n_revs * 360.0, dpsi_deg))

    beta_history = []
    beta_rate_history = []
    beta_accel_history = []
    lift_history = []

    dbeta_filtered = 0.0
    ddbeta_filtered = 0.0
    flap_state = [0.0, 0.0, 0.0]  # [beta, dbeta, ddbeta]

    current_time = 1.1
    body_rot = [0,0,0]
    # Precompute spanwise lever arms from hinge for least-squares modal extraction
    r_rel = tracked_x - r_root
    r_rel_sq_sum = np.sum(r_rel**2)

    for step, psi in enumerate(azimuth_steps):
        current_time += dt
        mapdl.time(current_time)

        # 1. Calculate & Distribute Aerodynamic Loads (with smooth 0.5-rev ramp)
        total_lift = 0.0
        t_ramp = min(1.0, step / (0.5 * steps_per_rev))
        ramp_factor = t_ramp * t_ramp * (3.0 - 2.0 * t_ramp)

        nodal_fy = np.zeros(len(tracked_nodes))
        nodal_fz = np.zeros(len(tracked_nodes))
        nodal_mx = np.zeros(len(tracked_nodes))

        for i, x_k in enumerate(x_centers):
            loads = get_sectional_aero_loads(ka25_data, body_vel, body_rot, flap_state, psi, x_k)

            F_lift = loads[2] * ramp_factor
            F_drag = loads[1] * ramp_factor
            Mx_pitch = loads[3] * ramp_factor

            total_lift += F_lift * dl

            # FIX: Split element load 50/50 between inner node (i) and outer node (i+1)
            nodal_fy[i]     += 0.5 * F_lift * dl
            nodal_fy[i + 1] += 0.5 * F_lift * dl
            nodal_fz[i]     += -0.5 * F_drag * dl
            nodal_fz[i + 1] += -0.5 * F_drag * dl
            nodal_mx[i]     += 0.5 * Mx_pitch * dl
            nodal_mx[i + 1] += 0.5 * Mx_pitch * dl

        
        with mapdl.non_interactive:
            for n_idx in range(1, len(tracked_nodes)):
                node_id = int(tracked_nodes[n_idx])
                mapdl.f(node_id, "FY", nodal_fy[n_idx])
                mapdl.f(node_id, "FZ", nodal_fz[n_idx])
                mapdl.f(node_id, "MX", nodal_mx[n_idx])
            
        # 2. Structural Solve
        mapdl.autots("ON")
        mapdl.kbc(0)
        mapdl.nsubst(2, 10, 1)
        mapdl.solve()
        
        # 3. Extract Implicit Kinematics 
        with mapdl.non_interactive:
            mapdl.run(f"*DO,I,1,{num_nodes}")
            mapdl.run("N_ID = NODE_LIST(I)")
            mapdl.run("*GET, UY_ALL(I), NODE, N_ID, U, Y")
            mapdl.run("*GET, VY_ALL(I), NODE, N_ID, V, Y")
            mapdl.run("*GET, AY_ALL(I), NODE, N_ID, A, Y")
            mapdl.run("*ENDDO")
        
        # Fetch the populated arrays back to Python in 3 fast calls
        uy_all = mapdl.parameters['UY_ALL'].flatten()
        vy_all = mapdl.parameters['VY_ALL'].flatten()
        ay_all = mapdl.parameters['AY_ALL'].flatten()
        
        # Least-squares rigid-body flapping projection across span (rejects high-freq tip noise)
        new_beta = np.arctan2(np.sum(r_rel * (uy_all - uy_all[0])), r_rel_sq_sum)
        dbeta_raw = np.sum(r_rel * (vy_all - vy_all[0])) / r_rel_sq_sum
        ddbeta_raw = np.sum(r_rel * (ay_all - ay_all[0])) / r_rel_sq_sum

        # Low-pass relaxation filter for partitioned FSI stability
        dbeta_filtered = 0.30 * dbeta_raw + 0.70 * dbeta_filtered
        ddbeta_filtered = 0.30 * ddbeta_raw + 0.70 * ddbeta_filtered

        beta_history.append(new_beta)
        beta_rate_history.append(dbeta_filtered)
        beta_accel_history.append(ddbeta_filtered)
        lift_history.append(total_lift)

        # Update complete flap state [beta, dbeta, ddbeta]
        flap_state = [new_beta, dbeta_filtered, ddbeta_filtered]

        if step % 18 == 0:
            print(
                f"  Step {step:3d} (Psi={np.degrees(psi):6.1f} deg): "
                f"Lift={total_lift:8.1f} N | "
                f"beta={np.degrees(new_beta):7.4f} deg | "
                f"dbeta={np.degrees(dbeta_filtered):8.3f} deg/s | "
                f"tipUY={uy_all[-1]:7.4f} m"
            )

        # COMMENT #2: Check Lift and Beta Convergence at the end of each revolution
        if (step + 1) % steps_per_rev == 0:
            rev_num = (step + 1) // steps_per_rev
            curr_beta_mean = np.degrees(np.mean(beta_history[-steps_per_rev:]))
            curr_lift_mean = np.mean(lift_history[-steps_per_rev:])
            if rev_num > 1:
                prev_beta_mean = np.degrees(np.mean(beta_history[-2 * steps_per_rev : -steps_per_rev]))
                prev_lift_mean = np.mean(lift_history[-2 * steps_per_rev : -steps_per_rev])
                d_beta = abs(curr_beta_mean - prev_beta_mean)
                d_lift_pct = 100.0 * abs(curr_lift_mean - prev_lift_mean) / max(abs(curr_lift_mean), 1e-6)
                print(
                    f"  >>> [REV {rev_num} CONVERGENCE] Mean Beta = {curr_beta_mean:.4f} deg "
                    f"(Delta = {d_beta:.4f} deg) | Mean Lift = {curr_lift_mean:.1f} N "
                    f"(Delta = {d_lift_pct:.3f}%)"
                )
            else:
                print(
                    f"  >>> [REV {rev_num} COMPLETE] Mean Beta = {curr_beta_mean:.4f} deg | "
                    f"Mean Lift = {curr_lift_mean:.1f} N"
                )
                
    # ==========================================
    # PHASE 4: STRESS EXTRACTION IN /POST1
    # ==========================================
    # FIX: Explicitly enter /POST1 before querying element SMISC records
    mapdl.post1()
    mapdl.set("LAST")

    my_list, mz_list, fx_list = [], [], []
    axial_stress_mpa, bending_stress_mpa, spanwise_stresses_mpa = [], [], []
    max_stress_pa = 0.0

    for i in range(1, n_elements + 1):
        fx_raw = mapdl.get_value("ELEM", i, "SMISC", 1)
        my_raw = mapdl.get_value("ELEM", i, "SMISC", 2)
        mz_raw = mapdl.get_value("ELEM", i, "SMISC", 3)

        Fx = float(fx_raw) if fx_raw is not None else 0.0
        My = float(my_raw) if my_raw is not None else 0.0
        Mz = float(mz_raw) if mz_raw is not None else 0.0

        axial_stress = abs(Fx) / Area
        k_y = (Mz * Iyy - My * Iyz) / denom
        k_z = (My * Izz - Mz * Iyz) / denom
        max_bending = abs(k_y * c_y) + abs(k_z * c_z)
        total_stress = axial_stress + max_bending

        my_list.append(My)
        mz_list.append(Mz)
        fx_list.append(Fx)
        axial_stress_mpa.append(axial_stress / 1e6)
        bending_stress_mpa.append(max_bending / 1e6)
        spanwise_stresses_mpa.append(total_stress / 1e6)

        if total_stress > max_stress_pa:
            max_stress_pa = total_stress

    max_stress_mpa = max_stress_pa / 1e6
    run_results.append({
            'Run': run_id + 1,
            'Velocity': vel,
            'RPM': rpm,
            'Pitch': pitch_deg,
            'Max_Stress_MPa': round(max_stress_mpa, 2)
        })
    
mapdl.exit()

print("\n=======================================================")
print(" DESIGN OF EXPERIMENTS (DoE) COMPLETE")
print("=======================================================")
df_results = pd.DataFrame(run_results)
print(df_results.to_string(index=False))
