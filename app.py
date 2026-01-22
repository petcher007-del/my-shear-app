import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings

# ปิดข้อความแจ้งเตือนที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="General Shear Assessment", layout="wide")

st.title("🏗️ Structural Shear Strength Assessment (Interactive Table)")
st.markdown("""
**Method:** Sigma-x Analysis (Analytical Model)  
*Flexible tool with interactive table for experimental data comparison.*
""")

# ==========================================
# 1. INPUTS (รับค่าตัวแปร)
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    # --- 1. Material Properties ---
    with st.expander("1. Material Properties", expanded=True):
        fc_prime = st.number_input("f'c (MPa)", value=31.5, step=0.5)
        default_Ec = 4700 * np.sqrt(fc_prime)
        Es = st.number_input("Es (MPa)", value=200000.0)
        Ec = default_Ec 

    # --- 2. Reinforcement ---
    with st.expander("2. Reinforcement", expanded=True):
        # สร้างคอลัมน์ให้ถูกต้อง (FIXED ERROR HERE)
        col_rho1, col_rho2 = st.columns(2)
        rho_l = col_rho1.number_input("rho_l", value=0.0264, format="%.4f")
        rho_v = col_rho2.number_input("rho_v", value=0.0029, format="%.4f")
        rho_h = col_rho1.number_input("rho_h", value=0.0029, format="%.4f")
        
        col_fy1, col_fy2 = st.columns(2)
        fy_l = col_fy1.number_input("fy_l (MPa)", value=438.0)
        fy_v = col_fy2.number_input("fy_v (MPa)", value=435.0)
        fy_h = col_fy1.number_input("fy_h (MPa)", value=435.0)

    # --- 3. Crack & Geometry ---
    with st.expander("3. Crack & Geometry", expanded=True):
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        s_cr = st.number_input("Crack Spacing (mm)", value=268.0)

    # --- 4. Boundary Conditions ---
    with st.expander("4. Boundary Conditions", expanded=False):
        use_clamping = st.checkbox("Apply Clamping Forces?", value=True)
        if use_clamping:
            h_av = st.number_input("h_av (mm)", value=1067.0)
            av = st.number_input("av (mm)", value=1767.0)
            x_cr1 = st.number_input("x_cr1 (mm)", value=681.0)
            x_cr2 = st.number_input("x_cr2 (mm)", value=655.0)
        else:
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0 

    # --- 5. Experimental Data (แบบตาราง Interactive) ---
    with st.expander("5. Experimental Data (Table)", expanded=True):
        st.write("📝 **กรอกข้อมูลผลการทดลอง:**")
        st.caption("ใส่ค่า 'กำลังที่เหลืออยู่' (Remaining Capacity) โปรแกรมจะแปลงเป็น %Loss ในกราฟให้เอง")
        
        # สร้างข้อมูลตั้งต้น (ตัวอย่างจาก Paper)
        default_data = pd.DataFrame([
            {"Width (mm)": 0.05, "Capacity (%)": 71.4},
            {"Width (mm)": 0.23, "Capacity (%)": 65.7},
            {"Width (mm)": 0.48, "Capacity (%)": 54.2},
            {"Width (mm)": 0.79, "Capacity (%)": 42.8},
            {"Width (mm)": 1.08, "Capacity (%)": 31.4},
            {"Width (mm)": 1.27, "Capacity (%)": 19.9},
            {"Width (mm)": 1.71, "Capacity (%)": 12.5},
            {"Width (mm)": 2.03, "Capacity (%)": 8.5},
        ])
        
        # สร้างตารางที่แก้ไขได้ (Data Editor)
        edited_df = st.data_editor(
            default_data, 
            num_rows="dynamic", # อนุญาตให้เพิ่ม/ลบแถวได้
            hide_index=True
        )
        
        plot_exp = st.checkbox("Plot Experimental Data", value=True)

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
    
    eps_x = eps_1*s2 + eps_2*c2 - gam_cr*sc
    eps_y = eps_1*c2 + eps_2*s2 + gam_cr*sc
    
    fc1 = (0.33 * np.sqrt(props['fc_prime'])) / (1 + np.sqrt(633 * eps_1))
    fc1 = min(fc1, 4.2)
    
    term = (-eps_1/(eps_2 if eps_2!=0 else 1e-9)) - 0.28
    beta_d = 1.0 if term < 0 else 1/(1+0.27*(term**0.8))
    if np.isnan(beta_d) or beta_d>1: beta_d=1.0
    
    ratio = eps_2 / (beta_d * -0.002)
    fc2 = 0 if ratio < 0 else -beta_d * props['fc_prime'] * (2*ratio - ratio**2)
    vci = 0 if (eps_1**2 + gam_cr**2)==0 else 3.83*(props['fc_prime']**(1/3))*(gam_cr**2/(eps_1**2+gam_cr**2))
    
    def fs(e, fy): return max(min(e*props['Es'], fy), -fy)
    
    sig_x = fc1*s2 + fc2*c2 - 2*vci*sc + props['rho_l']*fs(eps_x,props['fy_l']) + props['rho_h']*fs(eps_x,props['fy_h'])
    sig_y = fc1*c2 + fc2*s2 + 2*vci*sc + props['rho_v']*fs(eps_y,props['fy_v'])
    tau = (fc1-fc2)*sc + vci*(s2-c2)
    
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
    
    # --- A. Parse Data from Table ---
    w_exp, loss_exp = [], []
    has_exp_data = False
    
    if plot_exp and not edited_df.empty:
        try:
            # ดึงข้อมูลจากตาราง (DataFrame)
            w_exp = edited_df["Width (mm)"].tolist()
            cap_vals = edited_df["Capacity (%)"].tolist()
            
            # แปลง Remaining Capacity -> Loss (Degradation)
            # Loss = 100 - Remaining Capacity
            loss_exp = [100 - c for c in cap_vals]
            has_exp_data = True
        except Exception as e:
            st.error(f"Error reading table data: {e}")

    # --- B. Run Simulation ---
    w_range = np.linspace(0.05, 2.50, 50)
    tau_model = []
    curr = [-0.0001, 0.0002]
    
    progress_bar = st.progress(0)
    
    for i, w in enumerate(w_range):
        func = lambda x: obj_func(x, w/s_cr, props, theta_deg, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        
        if ier == 1:
            # Recalculate Tau logic
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
            curr = sol
        else:
            tau_model.append(np.nan)
        progress_bar.progress((i+1)/len(w_range))
    
    # Process Results
    tau_model = np.array(tau_model)
    tau_u = np.nanmax(tau_model)
    degradation = (1 - (tau_model/tau_u))*100
    
    # --- C. Plotting ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 1. Model Curve
        ax.plot(w_range, degradation, color='#d62728', linewidth=3, label='Analytical Model')
        
        # 2. Experimental Data
        if has_exp_data:
            ax.plot(w_exp, loss_exp, 'ro', markersize=8, markeredgecolor='k', label='User Data')
        
        ax.set_xlabel('Max Diagonal Crack Width, w_cr (mm)', fontweight='bold')
        ax.set_ylabel('Shear Strength Degradation (%)', fontweight='bold')
        ax.set_title(f'Shear Degradation Curve', fontsize=14)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.subheader("📊 Result Summary")
        st.metric("Max Shear Strength (Tau_u)", f"{tau_u:.2f} MPa")
        
        if has_exp_data:
            st.success("✅ User Data Loaded")
            with st.expander("ตรวจสอบข้อมูลที่พล็อต (Calculated Loss)"):
                # แสดงตารางผลลัพธ์ที่แปลงค่าแล้วให้ User ดู
                df_res = pd.DataFrame({
                    'Width (mm)': w_exp,
                    'Input Capacity (%)': edited_df["Capacity (%)"].tolist(),
                    'Calculated Loss (%)': loss_exp
                })
                st.dataframe(df_res, hide_index=True)
        else:
            st.info("No experimental data plotted.")

else:
    st.info("👈 Adjust parameters and click **Run Analysis**")
