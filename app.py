import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings
from PIL import Image # เพิ่ม library สำหรับจัดการรูปภาพ

# ปิดข้อความแจ้งเตือนที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="General Shear Assessment", layout="wide")

st.title("🏗️ Structural Shear Strength Assessment")
st.markdown("""
**Method:** Sigma-x Analysis (Analytical Model)  
*Flexible tool with interactive table and section image display.*
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

    # --- 5. Experimental Data (Table) ---
    with st.expander("5. Experimental Data (Table)", expanded=True):
        st.write("📝 **Experimental Data:**")
        st.caption("Input 'Degradation / Loss (%)' directly.")
        
        default_data = pd.DataFrame([
            {"Width (mm)": 0.05, "Loss (%)": 28.6},
            {"Width (mm)": 0.23, "Loss (%)": 34.3},
            {"Width (mm)": 0.48, "Loss (%)": 45.8},
            {"Width (mm)": 0.79, "Loss (%)": 57.2},
            {"Width (mm)": 1.08, "Loss (%)": 68.6},
            {"Width (mm)": 1.27, "Loss (%)": 80.1},
            {"Width (mm)": 1.71, "Loss (%)": 87.5},
            {"Width (mm)": 2.03, "Loss (%)": 91.5},
        ])
        
        edited_df = st.data_editor(default_data, num_rows="dynamic", hide_index=True)
        plot_exp = st.checkbox("Plot Experimental Data", value=True)

    # --- 6. Section Image Upload (NEW!) ---
    with st.expander("6. Cross-Section Image", expanded=True):
        st.write("📷 **Upload Section Drawing:**")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

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
    vci = 0 if (eps_1**2 + gam_cr**
