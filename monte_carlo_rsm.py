#MONTE CARLO WITH RESPONSE SURFACE MODEL
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. GROUND TRUTH DATA (From ANSYS FSI)
# ==========================================
# [Velocity, RPM, Pitch]
X_data = np.array([
    [15.0, 23.50, 10.0], [40.0, 23.50, 10.0],
    [15.0, 26.10, 10.0], [40.0, 26.10, 10.0],
    [15.0, 24.82,  8.0], [40.0, 24.82,  8.0],
    [15.0, 24.82, 12.0], [40.0, 24.82, 12.0],
    [27.5, 23.50,  8.0], [27.5, 26.10,  8.0],
    [27.5, 23.50, 12.0], [27.5, 26.10, 12.0],
    [27.5, 24.82, 10.0]  # Center point
])

# Resulting Max Stress (MPa)
Y_data = np.array([304.57, 314.89, 344.13, 351.4, 279.71, 285.67, 378.35, 385.81, 264.24, 300.8, 362.74, 403.23, 328.49])

# ==========================================
# 2. TRAIN THE RESPONSE SURFACE MODEL (RSM)
# ==========================================
# Define a 2nd-order polynomial function for 3 variables
def rsm_poly(X, c0, c1, c2, c3, c11, c22, c33, c12, c13, c23):
    v, r, p = X
    return (c0 + c1*v + c2*r + c3*p + 
            c11*v**2 + c22*r**2 + c33*p**2 + 
            c12*v*r + c13*v*p + c23*r*p)

# Transpose X_data to match curve_fit signature (3, 13)
popt, pcov = curve_fit(rsm_poly, X_data.T, Y_data)

# 3. MONTE CARLO SIMULATION (10,000 RUNS)
n_runs = 100000
vel_mc = np.random.normal(loc=27.5, scale=5.0, size=n_runs)   
rpm_mc = np.random.normal(loc=24.82, scale=0.5, size=n_runs)  
pitch_mc = np.random.normal(loc=10.0, scale=0.75, size=n_runs)


vel_mc = np.clip(vel_mc, 15.0, 40.0)
rpm_mc = np.clip(rpm_mc, 23.5, 26.1)
pitch_mc = np.clip(pitch_mc, 8.0, 12.0)


X_mc = np.vstack((vel_mc, rpm_mc, pitch_mc))
stresses_mc = rsm_poly(X_mc, *popt)

# 4. RELIABILITY ANALYSIS
FoS = 1.1
fatigue_limit_mpa = 450.0 / FoS
mean_stress = np.mean(stresses_mc)
max_sim_stress = np.max(stresses_mc)
p99_stress = np.percentile(stresses_mc, 99)
failures = np.sum(stresses_mc > fatigue_limit_mpa)
prob_failure = (failures / n_runs) * 100.0
RMSE1 = 0
for i in range(len(Y_data)):
    error = Y_data[i] - rsm_poly(X_data[i], *popt)
    RMSE1 += (error**2)
RMSE = np.sqrt(RMSE1/len(Y_data))
print("-"*45)
print("-"*45)
print(f"RMSE in curve_fit = {RMSE:.5f}")
print(f"Mean Operating Stress:   {mean_stress:.2f} MPa")
print(f"99th Percentile Stress:  {p99_stress:.2f} MPa")
print(f"Maximum Stress Hit:      {max_sim_stress:.2f} MPa")
print(f"Probability of Failure:  {prob_failure:.7f}%")
print("-"*45)
print("-"*45)
plt.figure(figsize=(10, 6))
plt.hist(stresses_mc, bins=60, color='royalblue', edgecolor='black', alpha=0.7)
plt.axvline(fatigue_limit_mpa, color='red', linestyle='--', linewidth=2.5, label=f'Fatigue Limit ({fatigue_limit_mpa} MPa)')
plt.axvline(mean_stress, color='black', linestyle='-.', linewidth=1.5, label=f'Mean Stress ({mean_stress:.1f} MPa)')

plt.title("Reliability Analysis", fontsize=14)
plt.xlabel("Maximum Root Stress (MPa)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 5. SENSITIVITY ANALYSIS (MAIN EFFECTS)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

v_sweep = np.linspace(15.0, 40.0, 50)
stress_v = rsm_poly((v_sweep, 24.82, 10.0), *popt)
axes[0].plot(v_sweep, stress_v, 'b-', linewidth=2.5)
axes[0].set(title="Wind Velocity Effect", xlabel="Velocity (m/s)", ylabel="Max Root Stress (MPa)")

r_sweep = np.linspace(23.5, 26.1, 50)
stress_r = rsm_poly((27.5, r_sweep, 10.0), *popt)
axes[1].plot(r_sweep, stress_r, 'g-', linewidth=2.5)
axes[1].set(title="Engine RPM Effect", xlabel="RPM (rad/s)")

p_sweep = np.linspace(8.0, 12.0, 50)
stress_p = rsm_poly((27.5, 24.82, p_sweep), *popt)
axes[2].plot(p_sweep, stress_p, 'r-', linewidth=2.5)
axes[2].set(title="Pilot Pitch Effect", xlabel="Collective Pitch (deg)")

for ax in axes: 
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle("Parameter Sensitivity Analysis on Rotor Root Cuff", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
