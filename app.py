import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings

# ปิดข้อความแจ้งเตือน
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="General Shear Assessment", layout="wide")

st.title("🏗️ General Crack-Based Shear Assessment")
st.markdown("""
**Method:** Fixed-Crack Continuum Modeling Approach (Refined based on Paper)  
**Flexibility:** Supports any beam geometry and optional experimental data comparison.
""")

# ==========================================
# 1. SIDEBAR INPUTS (รับค่าตัวแปร)
# ==========================================
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # --- 1. Material ---
    with st.expander("1. Material Properties", expanded=True):
        fc_prime = st.number_input("f'c (MPa)", value=31.5, step=0.5)
        Ec = st.number_input("Ec (MPa)", value=26385.0, help="Default is approx 4700sqrt(fc) or measured value") 
        Es = st.number_input("Es (MPa)", value=200000.0)
        
        st.caption("Tension Stiffening (Paper Eq. 10)")
        eps_cr = st.number_input("Reference Strain (eps_tu)", value=0.0002, format="%.5f")
        c_factor = st.number_input("Power Factor (c)", value=0.40)

    # --- 2. Reinforcement ---
    with st.expander("2. Reinforcement Ratios", expanded=True):
        col1, col2 = st.columns(2)
        rho_l = col1.number_input("rho_l", value=0.0264, format="%.4f")
        fy_l = col2.number_input("fy_l (MPa)", value=438.0)
        
        rho_v = col1.number_input("rho_v", value=0.0029, format="%.4f")
        fy_v = col2.number_input("fy_v (MPa)", value=435.0)
        
        rho_h = col1.number_input("rho_h", value=0.0029, format="%.4f")
        fy_h = col2.number_input("fy_h (MPa)", value=435.0)

    # --- 3. Geometry ---
    with st.expander("3. Crack Geometry", expanded=True):
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        s_cr = st.number_input("Crack Spacing (mm)", value=268.0)

    # --- 4. Boundary Conditions ---
    with st.expander("4. Boundary Conditions (Advanced)", expanded=False):
        use_clamping = st.checkbox("Enable Clamping Forces (Lab Setup)?", value=False)
        if use_clamping:
            st.caption("Geometry for clamping calculation:")
            h_av = st.number_input("h_av (mm)", value=1067.0)
            av = st.number_input("av (mm)", value=1767.0)
            x_cr1 = st.number_input("x_cr1 (mm)", value=681.0)
            x_cr2 = st.number_input("x_cr2 (mm)", value=655.0)
        else:
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0

    # --- 5. Experimental Data ---
    with st.expander("5. Experimental Data (Optional)", expanded=False):
        st.caption("Paste CSV: Width (mm), ResidualCapacity (%)")
        st.caption("Example: 0.05, 71.4")
        user_csv = st.text_area("Data Points", height=100)
        plot_exp = st.checkbox("Plot User Data", value=True)

# Pack variables
props = {
    'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 
    'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 
    'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h,
    'eps_cr': eps_cr, 'c_factor': c_factor
}
geom = {
    'h_av': h_av, 'av': av, 'x_cr1': x_cr1, 'x_cr2': x_cr2, 
    'use_clamping': use_clamping
}

# ==========================================
# 2. SOLVER LOGIC (Refined Physics)
# ==========================================
def obj_func(x, eps_1, props, theta_deg, geom):
    eps_2, gam_cr = x[0], x[1]
    
    # 1. Strain Transformation
    th = np.deg2rad(theta_deg)
    s, c = np.sin(th), np.cos(th)
    s2, c2, sc = s**2, c**2, s*c
    
    eps_x = eps_1*s2 + eps_2*c2 - gam_cr*sc
    eps_y = eps_1*c2 + eps_2*s2 + gam_cr*sc
    
    # 2. Concrete Models
    # Compression Softening (Vecchio & Collins style)
    if eps_2 == 0: beta_d = 1.0
    else:
        term = -eps_1/eps_2
        beta_d = 1.0 / (1.0 + 0.27 * (term - 0.28)**0.8) if term > 0.28 else 1.0
        if np.isnan(beta_d) or beta_d > 1: beta_d = 1.0
        if beta_d < 0.1: beta_d = 0.1
        
    epsc0 = -0.002
    ratio = eps_2 / (beta_d * epsc0)
    fc2 = -beta_d * props['fc_prime'] * (2*ratio - ratio**2) if (ratio > 0 and ratio <= 2) else 0
    
    # Tension Stiffening (Paper Eq. 10)
    f_cr = 0.33 * np.sqrt(props['fc_prime'])
    # Avoid division by zero or huge numbers at very small strains
    if eps_1 < 1e-6: 
        fc1 = props['Ec'] * eps_1
    else:
        # Power law decay
        fc1 = f_cr * (props['eps_cr'] / eps_1) ** props['c_factor']
        if fc1 > f_cr: fc1 = f_cr # Cap at cracking strength
        if fc1 < 0: fc1 = 0

    # Aggregate Interlock (Walraven Simplified)
    denom = eps_1**2 + gam_cr**2
    vci = 3.83 * (props['fc_prime']**(1/3)) * (gam_cr**2 / denom) if denom > 0 else 0
    
    # 3. Steel
    def get_fs(eps, fy): return max(min(eps * props['Es'], fy), -fy)
    fsl = get_fs(eps_x, props['fy_l'])
    fsv = get_fs(eps_y, props['fy_v'])
    fsh = get_fs(eps_x, props['fy_h'])
    
    # 4. Equilibrium
    sig_x = fc1*s2 + fc2*c2 - 2*vci*sc + props['rho_l']*fsl + props['rho_h']*fsh
    sig_y = fc1*c2 + fc2*s2 + 2*vci*sc + props['rho_v']*fsv
    tau   = (fc1 - fc2)*sc + vci*(s2 - c2)
    
    # 5. Residuals
    res1 = sig_x # Sigma_x should be 0
    
    if geom['use_clamping']:
        # Clamping Logic
        c1, c2 = 1417, 1394
        t1 = 2.5/(0.6+4*(geom['x_cr1']/c1))-0.5
        t2 = 2.5/(0.6+4*(geom['x_cr2']/c2))-0.5
        tgt = -0.5*(geom['h_av']/geom['av'])*(t1+t2)
        cur = sig_y / tau if abs(tau) > 1e-4 else 0
        res2 = cur - tgt
    else:
        # General Beam Logic (Sigma_y = 0)
        res2 = sig_y
        
    return [res1, res2]

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if st.button("🚀 Run General Analysis", type="primary"):
    
    # --- A. Parse User Data (if any) ---
    w_exp, cap_exp = [], []
    has_exp_data = False
    
    if plot_exp and user_csv.strip():
        try:
            lines = user_csv.strip().split('\n')
            for line in lines:
                p = line.split(',')
                if len(p) >= 2:
                    w_exp.append(float(p[0]))
                    cap_exp.append(float(p[1])) # Expecting Remaining Capacity %
            has_exp_data = True
        except:
            st.error("Invalid CSV format. Please check your data.")

    # --- B. Run Simulation ---
    w_range = np.linspace(0.05, 2.50, 50)
    tau_model = []
    curr = [-0.0002, 0.0005] # Initial guess
    
    prog = st.progress(0)
    
    for i, w in enumerate(w_range):
        func = lambda x: obj_func(x, w/s_cr, props, theta_deg, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        
        if ier == 1:
            # Re-calculate Tau for result
            # (Repeating physics calculation for extraction)
            eps_2, gam_cr = sol[0], sol[1]
            th = np.deg2rad(theta_deg); s,c=np.sin(th),np.cos(th); s2,c2,sc=s**2,c**2,s*c
            
            # Recalc components
            term = - (w/s_cr)/eps_2
            bd = 1.0/(1+0.27*(term-0.28)**0.8) if term>0.28 else 1.0
            r = eps_2/(bd*-0.002)
            fc2 = -bd*props['fc_prime']*(2*r-r**2) if (r>0 and r<=2) else 0
            
            f_cr = 0.33*np.sqrt(props['fc_prime'])
            fc1 = f_cr * (props['eps_cr']/(w/s_cr))**props['c_factor']
            if fc1 > f_cr: fc1 = f_cr
            
            vci = 3.83*(props['fc_prime']**(1/3))*(gam_cr**2/((w/s_cr)**2+gam_cr**2))
            tau = (fc1-fc2)*sc + vci*(s2-c2)
            
            tau_model.append(tau)
            curr = sol
        else:
            tau_model.append(np.nan)
        prog.progress((i+1)/len(w_range))
        
    # --- C. Process & Plot ---
    tau_model = np.array(tau_model)
    tau_u = np.nanmax(tau_model)
    # Convert to Remaining Capacity % (Matching the Paper's Y-axis)
    # If you want Degradation (Loss), use (1 - ratio)*100
    # Paper uses "Residual Shear Capacity" which decreases
    residual_pct = (tau_model / tau_u) * 100
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 1. Model Curve
        ax.plot(w_range, residual_pct, 'b-', lw=3, label='Analytical Model')
        
        # 2. Experimental Data (if provided)
        if has_exp_data:
            ax.plot(w_exp, cap_exp, 'rd', markersize=8, markeredgecolor='k', label='User Data')
            
        ax.set_xlabel('Max Diagonal Crack Width (mm)', fontweight='bold')
        ax.set_ylabel('Residual Shear Capacity (%)', fontweight='bold')
        ax.set_title(f'Shear Capacity Assessment (Tau_u = {tau_u:.2f} MPa)')
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("📊 Summary")
        st.metric("Max Shear Strength", f"{tau_u:.2f} MPa")
        
        if has_exp_data:
            st.success("✅ Experimental data plotted.")
            df = pd.DataFrame({"Width": w_exp, "Capacity(%)": cap_exp})
            with st.expander("View Data"):
                st.dataframe(df)
        else:
            st.info("ℹ️ No experimental data provided. (You can paste CSV in the sidebar)")

else:
    st.info("👈 Adjust parameters and click **Run General Analysis**")
