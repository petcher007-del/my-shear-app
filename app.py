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

st.title("🏗️ Structural Shear Safety Assessment (General Case)")
st.markdown("**Method:** Sigma-x Analysis (Original Logic)")

# ==========================================
# 1. SIDEBAR INPUTS (รับค่าตัวแปร)
# ==========================================
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # --- 1. Material ---
    with st.expander("1. Material Properties", expanded=True):
        fc_prime = st.number_input("f'c (MPa)", value=31.5, step=0.5)
        # คำนวณ Ec อัตโนมัติแบบเดิม
        default_Ec = 4700 * np.sqrt(fc_prime)
        Es = st.number_input("Es (MPa)", value=200000.0)
        Ec = default_Ec 

    # --- 2. Reinforcement ---
    with st.expander("2. Reinforcement", expanded=True):
        col1, col2 = st.columns(2)
        rho_l = col1.number_input("rho_l", value=0.0264, format="%.4f")
        fy_l = col2.number_input("fy_l (MPa)", value=438.0)
        
        rho_v = col1.number_input("rho_v", value=0.0029, format="%.4f")
        fy_v = col2.number_input("fy_v (MPa)", value=435.0)
        
        rho_h = col1.number_input("rho_h", value=0.0029, format="%.4f")
        fy_h = col2.number_input("fy_h (MPa)", value=435.0)

    # --- 3. Crack Geometry ---
    with st.expander("3. Crack Geometry", expanded=True):
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        s_cr = st.number_input("Crack Spacing (mm)", value=268.0)

    # --- 4. Boundary Conditions ---
    with st.expander("4. Boundary Conditions (Advanced)", expanded=False):
        use_clamping = st.checkbox("Enable Clamping Forces?", value=True)
        if use_clamping:
            h_av = st.number_input("h_av (mm)", value=1067.0)
            av = st.number_input("av (mm)", value=1767.0)
            x_cr1 = st.number_input("x_cr1 (mm)", value=681.0)
            x_cr2 = st.number_input("x_cr2 (mm)", value=655.0)
        else:
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0

    # --- 5. Experimental Data ---
    with st.expander("5. Experimental Data (Optional)", expanded=False):
        st.caption("Paste CSV: Width, ResidualCapacity(%)")
        # Default data from your example
        default_csv = "0.05, 71.4\n0.23, 65.7\n0.48, 54.2\n0.79, 42.8\n1.08, 31.4\n1.27, 19.9\n1.71, 12.5\n2.03, 8.5"
        user_csv = st.text_area("Data Points", value=default_csv, height=150)
        plot_exp = st.checkbox("Plot Experimental Data", value=True)

# Pack variables
props = {'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h}
geom = {'h_av': h_av, 'av': av, 'x_cr1': x_cr1, 'x_cr2': x_cr2, 'use_clamping': use_clamping}

# ==========================================
# 2. LOGIC (Original Function)
# ==========================================
def obj_func(x, eps_1, props, theta_deg, geom):
    # (ใช้ Logic เดิมที่เคยทำงานได้ถูกต้อง)
    eps_2, gam_cr = x[0], x[1]
    th = np.deg2rad(theta_deg)
    s, c = np.sin(th), np.cos(th)
    s2, c2, sc = s**2, c**2, s*c
    
    eps_x = eps_1*s2 + eps_2*c2 - gam_cr*sc
    eps_y = eps_1*c2 + eps_2*s2 + gam_cr*sc
    
    # Concrete Tension (Original Formula)
    fc1 = (0.33 * np.sqrt(props['fc_prime'])) / (1 + np.sqrt(633 * eps_1))
    fc1 = min(fc1, 4.2)
    
    # Concrete Compression (Original Formula)
    term = (-eps_1/(eps_2 if eps_2!=0 else 1e-9)) - 0.28
    beta_d = 1.0 if term < 0 else 1/(1+0.27*(term**0.8))
    if np.isnan(beta_d) or beta_d>1: beta_d=1.0
    
    ratio = eps_2 / (beta_d * -0.002)
    fc2 = 0 if ratio < 0 else -beta_d * props['fc_prime'] * (2*ratio - ratio**2)
    
    # Shear Transfer (Original Formula)
    vci = 0 if (eps_1**2 + gam_cr**2)==0 else 3.83*(props['fc_prime']**(1/3))*(gam_cr**2/(eps_1**2+gam_cr**2))
    
    def fs(e, fy): return max(min(e*props['Es'], fy), -fy)
    
    sig_x = fc1*s2 + fc2*c2 - 2*vci*sc + props['rho_l']*fs(eps_x,props['fy_l']) + props['rho_h']*fs(eps_x,props['fy_h'])
    sig_y = fc1*c2 + fc2*s2 + 2*vci*sc + props['rho_v']*fs(eps_y,props['fy_v'])
    tau = (fc1-fc2)*sc + vci*(s2-c2)
    
    # Boundary Conditions
    if geom['use_clamping']:
        c1, c2 = 1417, 1394
        t1 = 2.5/(0.6+4*(geom['x_cr1']/c1))-0.5
        t2 = 2.5/(0.6+4*(geom['x_cr2']/c2))-0.5
        tgt = -0.5*(geom['h_av']/geom['av'])*(t1+t2)
        cur = 0 if abs(tau)<1e-4 else sig_y/tau
        return [sig_x, cur-tgt]
    else:
        return [sig_x, sig_y]

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if st.button("🚀 Run Analysis", type="primary"):
    
    # A. Parse User Data
    w_exp, cap_exp = [], []
    has_exp = False
    if plot_exp and user_csv.strip():
        try:
            lines = user_csv.strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    w_exp.append(float(parts[0]))
                    cap_exp.append(float(parts[1]))
            has_exp = True
        except:
            st.error("CSV Format Error")

    # B. Run Simulation
    w_range = np.linspace(0.05, 2.50, 40)
    tau_model = []
    curr = [-0.0001, 0.0002]
    
    prog = st.progress(0)
    
    for i, w in enumerate(w_range):
        func = lambda x: obj_func(x, w/s_cr, props, theta_deg, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        if ier == 1:
            # Recalculate Tau logic (Original)
            th=np.deg2rad(theta_deg); s,c=np.sin(th),np.cos(th); s2,c2,sc=s**2,c**2,s*c
            
            fc1 = (0.33*np.sqrt(props['fc_prime']))/(1+np.sqrt(633*(w/s_cr))); fc1=min(fc1,4.2)
            
            term=(-(w/s_cr)/(sol[0] if sol[0]!=0 else 1e-9))-0.28
            bd=1.0 if term<0 else 1/(1+0.27*(term**0.8))
            if bd>1: bd=1.0
            
            r=sol[0]/(bd*-0.002); fc2=0 if r<0 else -bd*props['fc_prime']*(2*r-r**2)
            vci=3.83*(props['fc_prime']**(1/3))*(sol[1]**2/((w/s_cr)**2+sol[1]**2))
            
            tau=(fc1-fc2)*sc+vci*(s2-c2)
            tau_model.append(tau)
            curr = sol
        else:
            tau_model.append(np.nan)
        prog.progress((i+1)/len(w_range))
    
    # C. Plot
    tau_model = np.array(tau_model)
    tau_u = np.nanmax(tau_model)
    
    # แปลงเป็น Remaining Capacity (%) เพื่อเทียบกับกราฟใน Paper
    # ถ้าค่าโมเดลเป็น stress (MPa) เราจะ normalize ด้วย tau_u
    model_pct = (tau_model / tau_u) * 100
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.plot(w_range, model_pct, 'b-', lw=3, label='Estimated RSC (Model)')
        
        if has_exp:
            ax.plot(w_exp, cap_exp, 'rd', markeredgecolor='k', label='Measured RSC')
            
        ax.set_xlabel('Max Diagonal Crack Width (mm)')
        ax.set_ylabel('Residual Shear Capacity (%)')
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Results")
        st.metric("Max Capacity (Tau_u)", f"{tau_u:.2f} MPa")
        if has_exp:
            st.write("User Data Points:", pd.DataFrame({'Width': w_exp, 'Cap%': cap_exp}))
else:
    st.info("Click button to run")
