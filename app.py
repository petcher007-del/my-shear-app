import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

# ปิดข้อความแจ้งเตือนที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shear Assessment Model", layout="wide")

st.title("🏗️ Structural Shear Strength Assessment")
st.markdown("""
**Method:** Sigma-x Analysis (Analytical Model)  
*Predicts the shear strength degradation curve based on structural parameters.*
""")

# ==========================================
# 1. INPUTS (รับค่าตัวแปร)
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    with st.expander("1. Material Properties", expanded=True):
        fc_prime = st.number_input("f'c (MPa)", value=31.5, step=0.5)
        # คำนวณ Ec อัตโนมัติ (เปลี่ยนสูตรได้ถ้าต้องการ)
        default_Ec = 4700 * np.sqrt(fc_prime)
        Es = st.number_input("Es (MPa)", value=200000.0)
        Ec = default_Ec 

    with st.expander("2. Reinforcement", expanded=True):
        col_rho1, col_rho2 = st.columns(2)
        rho_l = col_rho1.number_input("rho_l", value=0.0264, format="%.4f")
        rho_v = col_rho2.number_input("rho_v", value=0.0029, format="%.4f")
        rho_h = col_rho1.number_input("rho_h", value=0.0029, format="%.4f")
        
        col_fy1, col_fy2 = st.columns(2)
        fy_l = col_fy1.number_input("fy_l (MPa)", value=438.0)
        fy_v = col_fy2.number_input("fy_v (MPa)", value=435.0)
        fy_h = col_fy1.number_input("fy_h (MPa)", value=435.0)

    with st.expander("3. Crack & Geometry", expanded=True):
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        s_cr = st.number_input("Crack Spacing (mm)", value=268.0)

    with st.expander("4. Boundary Conditions", expanded=False):
        st.caption("Enable this for beams with vertical clamping stress (e.g., special lab setup).")
        use_clamping = st.checkbox("Apply Clamping Forces?", value=True)
        
        if use_clamping:
            h_av = st.number_input("h_av (mm)", value=1067.0)
            av = st.number_input("av (mm)", value=1767.0)
            x_cr1 = st.number_input("x_cr1 (mm)", value=681.0)
            x_cr2 = st.number_input("x_cr2 (mm)", value=655.0)
        else:
            # ไม่คิดผลของ Clamping (Sigma_y Target = 0)
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0 

# Pack variables
props = {'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h}
geom = {'h_av': h_av, 'av': av, 'x_cr1': x_cr1, 'x_cr2': x_cr2, 'use_clamping': use_clamping}


# ==========================================
# 2. SOLVER LOGIC
# ==========================================
def obj_func(x, eps_1, props, theta_deg, geom):
    eps_2, gam_cr = x[0], x[1]
    th = np.deg2rad(theta_deg)
    s, c = np.sin(th), np.cos(th)
    s2, c2, sc = s**2, c**2, s*c
    
    # Strain Transformation
    eps_x = eps_1*s2 + eps_2*c2 - gam_cr*sc
    eps_y = eps_1*c2 + eps_2*s2 + gam_cr*sc
    
    # Constitutive Models
    fc1 = (0.33 * np.sqrt(props['fc_prime'])) / (1 + np.sqrt(633 * eps_1))
    fc1 = min(fc1, 4.2)
    
    term = (-eps_1/(eps_2 if eps_2!=0 else 1e-9)) - 0.28
    beta_d = 1.0 if term < 0 else 1/(1+0.27*(term**0.8))
    if np.isnan(beta_d) or beta_d>1: beta_d=1.0
    
    ratio = eps_2 / (beta_d * -0.002)
    fc2 = 0 if ratio < 0 else -beta_d * props['fc_prime'] * (2*ratio - ratio**2)
    
    vci = 0 if (eps_1**2 + gam_cr**2)==0 else 3.83*(props['fc_prime']**(1/3))*(gam_cr**2/(eps_1**2+gam_cr**2))
    
    def fs(e, fy): return max(min(e*props['Es'], fy), -fy)
    
    # Equilibrium
    sig_x = fc1*s2 + fc2*c2 - 2*vci*sc + props['rho_l']*fs(eps_x,props['fy_l']) + props['rho_h']*fs(eps_x,props['fy_h'])
    sig_y = fc1*c2 + fc2*s2 + 2*vci*sc + props['rho_v']*fs(eps_y,props['fy_v'])
    tau = (fc1-fc2)*sc + vci*(s2-c2)
    
    # Boundary Condition Logic
    if geom['use_clamping']:
        c1, c2 = 1417, 1394
        t1 = 2.5/(0.6+4*(geom['x_cr1']/c1))-0.5
        t2 = 2.5/(0.6+4*(geom['x_cr2']/c2))-0.5
        ratio_target = -0.5*(geom['h_av']/geom['av'])*(t1+t2)
        cur_ratio = 0 if abs(tau)<1e-4 else sig_y/tau
        return [sig_x, cur_ratio - ratio_target]
    else:
        return [sig_x, sig_y] # General Beam: Sigma_x = 0, Sigma_y = 0

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if st.button("🚀 Calculate Model", type="primary"):
    
    # สร้างช่วงความกว้างรอยร้าวที่จะคำนวณ (0.05 มม. ถึง 2.5 มม.)
    w_range = np.linspace(0.05, 2.50, 50)
    tau_model = []
    curr = [-0.0001, 0.0002] # Initial Guess
    
    progress_bar = st.progress(0)
    
    # Loop คำนวณทีละจุด
    for i, w in enumerate(w_range):
        func = lambda x: obj_func(x, w/s_cr, props, theta_deg, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        
        if ier == 1:
            # Recalculate Tau for plotting
            eps_2, gam_cr = sol[0], sol[1]
            th=np.deg2rad(theta_deg); s,c=np.sin(th),np.cos(th); s2,c2,sc=s**2,c**2,s*c
            fc1 = (0.33*np.sqrt(props['fc_prime']))/(1+np.sqrt(633*(w/s_cr))); fc1=min(fc1,4.2)
            term=(-(w/s_cr)/(sol[0] if sol[0]!=0 else 1e-9))-0.28
            bd=1.0 if term<0 else 1/(1+0.27*(term**0.8))
            if bd>1: bd=1.0
            r=sol[0]/(bd*-0.002); fc2=0 if r<0 else -bd*props['fc_prime']*(2*r-r**2)
            vci=3.83*(props['fc_prime']**(1/3))*(sol[1]**2/((w/s_cr)**2+sol[1]**2))
            tau=(fc1-fc2)*sc+vci*(s2-c2)
            
            tau_model.append(tau)
            curr = sol # Update guess for next step
        else:
            tau_model.append(np.nan)
        progress_bar.progress((i+1)/len(w_range))
    
    # Process Results
    tau_model = np.array(tau_model)
    tau_u = np.nanmax(tau_model)
    degradation = (1 - (tau_model/tau_u))*100
    
    # Plotting
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot Model Curve ONLY
        ax.plot(w_range, degradation, color='#d62728', linewidth=3, label='Analytical Model')
        
        ax.set_xlabel('Max Diagonal Crack Width, w_cr (mm)', fontweight='bold')
        ax.set_ylabel('Shear Strength Degradation (%)', fontweight='bold')
        ax.set_title(f'Predicted Shear Degradation Curve', fontsize=14)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.subheader("📊 Result")
        st.metric("Max Shear Strength", f"{tau_u:.2f} MPa")
        st.info("""
        **Interpretation:**
        The graph shows how much shear capacity is lost (Degradation %) as the crack width increases.
        """)

else:
    st.info("👈 Adjust parameters in the sidebar and click **Calculate Model**")
