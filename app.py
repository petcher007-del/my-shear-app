import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings

# ปิดข้อความแจ้งเตือนที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shear Assessment (Paper Refined)", layout="wide")

st.title("🏗️ Crack-Based Shear Strength Assessment")
st.markdown("""
**Reference:** *Crack-Based Shear Strength Assessment of Reinforced Concrete Members Using a Fixed-Crack Continuum Modeling Approach* **Beam ID:** DS3-42-1.85-03 (and General Case)
""")

# ==========================================
# 1. SIDEBAR INPUTS (รับค่าตัวแปร)
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    with st.expander("1. Material Properties", expanded=True):
        # Default values from Paper Table 1
        fc_prime = st.number_input("Concrete Compressive Strength, f'c (MPa)", value=31.5, step=0.5)
        # Ec estimate: 4700sqrt(fc) is standard ACI, keep unless specified otherwise
        Ec = st.number_input("Concrete Modulus, Ec (MPa)", value=26385.0) 
        Es = st.number_input("Steel Modulus, Es (MPa)", value=200000.0)
        
        st.markdown("---")
        st.caption("Tension Stiffening Parameters (Eq. 10)")
        epsilon_cr = st.number_input("Cracking Strain (eps_cr)", value=0.00008, format="%.5f")
        c_factor = st.number_input("Stiffening Parameter (c)", value=0.40)

    with st.expander("2. Reinforcement", expanded=True):
        # Default values from Paper Table 1
        st.caption("Longitudinal Steel")
        rho_l = st.number_input("rho_l", value=0.0264, format="%.4f")
        fy_l = st.number_input("fy_l (MPa)", value=438.0)
        
        st.caption("Transverse (Stirrups) Steel")
        rho_v = st.number_input("rho_v", value=0.0029, format="%.4f")
        fy_v = st.number_input("fy_v (MPa)", value=435.0)
        
        st.caption("Horizontal Web Steel (if any)")
        rho_h = st.number_input("rho_h", value=0.0029, format="%.4f")
        fy_h = st.number_input("fy_h (MPa)", value=435.0)

    with st.expander("3. Crack Geometry", expanded=True):
        # From Table 2
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        s_cr = st.number_input("Crack Spacing (mm)", value=268.0)

    with st.expander("4. Analysis Options", expanded=False):
        # Toggle for Lab Setup vs General Beam
        analysis_mode = st.radio(
            "Boundary Condition:",
            ("General Beam (Sigma_y = 0)", "Paper Experiment (Clamping Forces)")
        )
        
        if analysis_mode == "Paper Experiment (Clamping Forces)":
            st.info("Uses specific geometry from the paper's test setup.")
            h_av = 1067
            av = 1767
            x_cr1 = 681
            x_cr2 = 655
        else:
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0

# Pack variables for solver
props = {
    'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 
    'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 
    'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h,
    'eps_cr': epsilon_cr, 'c_factor': c_factor
}
geom = {
    'h_av': h_av, 'av': av, 'x_cr1': x_cr1, 'x_cr2': x_cr2, 
    'mode': analysis_mode
}

# ==========================================
# 2. SOLVER LOGIC (Refined Equations)
# ==========================================
def obj_func(x, eps_1, props, theta_deg, geom):
    eps_2, gam_cr = x[0], x[1]
    
    # 1. Strain Transformation [Eq. 1-3]
    th = np.deg2rad(theta_deg)
    s, c = np.sin(th), np.cos(th)
    s2, c2, sc = s**2, c**2, s*c
    
    eps_x = eps_1*s2 + eps_2*c2 - gam_cr*sc
    eps_y = eps_1*c2 + eps_2*s2 + gam_cr*sc
    
    # 2. Concrete Constitutive Models
    # 2.1 Compression (Softened) [Eq. 14, 15]
    # Parabolic stress-strain with softening factor beta_d (zeta)
    if eps_2 == 0:
        beta_d = 1.0
    else:
        # Vecchio & Collins 1986 Model or similar as implied by context
        # Standard softening factor
        term = -eps_1/eps_2
        if term < 0: term = 0
        beta_d = 1.0 / (1.0 + 0.27 * (term - 0.28)**0.8)
        if np.isnan(beta_d) or beta_d > 1: beta_d = 1.0
        if beta_d < 0.1: beta_d = 0.1 # Minimum cap
    
    # Peak strain usually -0.002
    epsc0 = -0.002
    ratio = eps_2 / (beta_d * epsc0)
    
    if ratio < 0: fc2 = 0
    elif ratio <= 2:
        fc2 = -beta_d * props['fc_prime'] * (2*ratio - ratio**2)
    else:
        fc2 = 0 # Crushed
        
    # 2.2 Tension Stiffening [Eq. 10 Refined]
    # Paper uses power law: f_c1 = f_cr * (eps_tu / eps_1)^c
    f_cr = 0.33 * np.sqrt(props['fc_prime']) # Tensile strength estimate
    if eps_1 <= props['eps_cr']:
        fc1 = props['Ec'] * eps_1
    else:
        # Use Paper's Eq 10 format
        # Note: eps_tu is not explicitly constant, usually relative to cracking
        # Adapting to typical tension stiffening form:
        fc1 = f_cr * (props['eps_cr'] / eps_1) ** props['c_factor']
        if fc1 < 0: fc1 = 0
    
    # 2.3 Shear Transfer (Aggregate Interlock) [Eq. 21]
    # Walraven / Li & Maekawa Model
    if (eps_1**2 + gam_cr**2) == 0:
        vci = 0
    else:
        # Simplified Contact Density Model form
        vci = 3.83 * (props['fc_prime']**(1/3)) * (gam_cr**2 / (eps_1**2 + gam_cr**2))
    
    # 3. Steel Stresses (Elasto-Plastic)
    def get_fs(eps, fy, Es):
        val = eps * Es
        return max(min(val, fy), -fy)
        
    fsl = get_fs(eps_x, props['fy_l'], props['Es'])
    fsv = get_fs(eps_y, props['fy_v'], props['Es'])
    fsh = get_fs(eps_x, props['fy_h'], props['Es'])
    
    # 4. Equilibrium Equations [Eq. 4-6]
    # Sigma_x_calc (Internal Force)
    sig_x_calc = fc1*s2 + fc2*c2 - 2*vci*sc + props['rho_l']*fsl + props['rho_h']*fsh
    
    # Sigma_y_calc (Internal Force)
    sig_y_calc = fc1*c2 + fc2*s2 + 2*vci*sc + props['rho_v']*fsv
    
    # Tau_xy_calc (Internal Shear Capacity)
    tau_xy = (fc1 - fc2)*sc + vci*(s2 - c2)
    
    # 5. Residual Definitions (Solver Targets)
    # Target 1: Sigma_x should be 0 (No external axial force)
    res1 = sig_x_calc 
    
    # Target 2: Depends on Boundary Condition
    if geom['mode'] == "Paper Experiment (Clamping Forces)":
        # Specific Clamping Logic (from previous code/setup)
        c1, c2 = 1417, 1394
        t1 = 2.5/(0.6+4*(geom['x_cr1']/c1))-0.5
        t2 = 2.5/(0.6+4*(geom['x_cr2']/c2))-0.5
        # Prevent div by zero
        tau_denom = tau_xy if abs(tau_xy) > 1e-4 else 1e-4
        current_ratio = sig_y_calc / tau_denom
        target_ratio = -0.5*(geom['h_av']/geom['av'])*(t1+t2)
        res2 = current_ratio - target_ratio
    else:
        # General Beam: Sigma_y should be 0 (Free expansion vertically)
        res2 = sig_y_calc
        
    return [res1, res2]

# ==========================================
# 3. MAIN APP EXECUTION
# ==========================================
if st.button("🚀 Calculate Shear Degradation", type="primary"):
    
    # Range of crack widths to analyze
    w_range = np.linspace(0.05, 2.50, 40)
    tau_results = []
    
    # Initial Guess for solver [eps_2, gam_cr]
    current_guess = [-0.0005, 0.001]
    
    prog_bar = st.progress(0)
    
    for i, w in enumerate(w_range):
        # Convert width to strain: eps_1 = w / s_cr
        eps_1_val = w / s_cr
        
        # Define wrapper for fsolve
        func = lambda x: obj_func(x, eps_1_val, props, theta_deg, geom)
        
        # Run Solver
        sol, info, ier, msg = fsolve(func, current_guess, full_output=True)
        
        if ier == 1:
            # If converged, extract Tau from the solution
            # Need to re-run specific parts of obj_func to get tau_xy
            # (Quick re-calc for plotting)
            x_final = sol
            eps_2, gam_cr = x_final[0], x_final[1]
            th = np.deg2rad(theta_deg)
            s, c = np.sin(th), np.cos(th)
            s2, c2, sc = s**2, c**2, s*c
            
            # Re-eval Constitutive (Simplified for display)
            # Note: Ideally refactor calc logic to separate function to avoid duplicate code
            vci = 3.83 * (props['fc_prime']**(1/3)) * (gam_cr**2 / (eps_1_val**2 + gam_cr**2))
            
            # Approximate fc1, fc2 again just for Tau calc
            # (Using simplified check for speed)
            term = -eps_1_val/eps_2
            bd = 1.0/(1+0.27*(term-0.28)**0.8) if term > 0.28 else 1.0
            ratio = eps_2 / (bd * -0.002)
            fc2 = -bd * props['fc_prime'] * (2*ratio - ratio**2) if ratio > 0 else 0
            
            f_cr = 0.33 * np.sqrt(props['fc_prime'])
            fc1 = f_cr * (props['eps_cr']/eps_1_val)**props['c_factor']
            
            tau_val = (fc1 - fc2)*sc + vci*(s2 - c2)
            tau_results.append(tau_val)
            
            # Update guess for next step (Continuation method)
            current_guess = sol
        else:
            tau_results.append(np.nan)
        
        prog_bar.progress((i+1)/len(w_range))
    
    # Process Data
    tau_results = np.array(tau_results)
    tau_max = np.nanmax(tau_results)
    degradation = (1 - (tau_results / tau_max)) * 100
    
    # Plotting
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(w_range, degradation, color='blue', linewidth=3, label='Estimated RSC (Model)')
        
        # Add labels matching the paper style
        ax.set_xlabel('Maximum Diagonal Crack Width (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residual Shear Capacity (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Degradation Curve (Tau_max = {tau_max:.2f} MPa)', fontsize=14)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Analysis Summary")
        st.metric("Max Shear Capacity", f"{tau_max:.2f} MPa")
        st.markdown(f"""
        **Parameters Used:**
        - f'c: {fc_prime} MPa
        -
