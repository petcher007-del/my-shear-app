import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

# ปิดข้อความแจ้งเตือนที่ไม่จำเป็น
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shear Assessment", layout="wide")

st.title("🏗️ Structural Shear Safety Assessment")
st.markdown("**Method:** Sigma-x Analysis for Reinforced Concrete Beams")

# --- ส่วนที่ 1: แถบด้านซ้าย (รับค่าตัวแปร) ---
st.sidebar.header("1. Material Properties")
fc_prime = st.sidebar.number_input("f'c (MPa)", value=31.5, step=0.5)
Ec = 4700 * np.sqrt(fc_prime)
Es = st.sidebar.number_input("Es (MPa)", value=200000.0)

st.sidebar.subheader("Reinforcement")
rho_l = st.sidebar.number_input("rho_l", value=0.0264, format="%.4f")
rho_v = st.sidebar.number_input("rho_v", value=0.0029, format="%.4f")
rho_h = st.sidebar.number_input("rho_h", value=0.0029, format="%.4f")
fy_l = st.sidebar.number_input("fy_l (MPa)", value=438.0)
fy_v = st.sidebar.number_input("fy_v (MPa)", value=435.0)
fy_h = st.sidebar.number_input("fy_h (MPa)", value=435.0)

st.sidebar.header("2. Crack Geometry")
theta_deg = st.sidebar.number_input("Crack Angle (deg)", value=46.0)
s_cr = st.sidebar.number_input("Crack Spacing (mm)", value=268.0)

# ค่าคงที่รูปทรงคาน
props = {'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h}
geom = {'h_av': 1067, 'av': 1767, 'x_cr1': 681, 'x_cr2': 655}

# --- ส่วนที่ 2: ฟังก์ชันคำนวณ (Logic) ---
def obj_func(x, eps_1, props, theta_deg, geom):
    # (ฟังก์ชันคำนวณย่อ)
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
    
    c1, c2 = 1417, 1394
    t1 = 2.5/(0.6+4*(geom['x_cr1']/c1))-0.5
    t2 = 2.5/(0.6+4*(geom['x_cr2']/c2))-0.5
    tgt = -0.5*(geom['h_av']/geom['av'])*(t1+t2)
    cur = 0 if abs(tau)<1e-4 else sig_y/tau
    return [sig_x, cur-tgt]

# --- ส่วนที่ 3: ปุ่มกดและแสดงผล ---
if st.button("🚀 Run Analysis"):
    w_exp = np.array([0.05, 0.23, 0.48, 0.79, 1.08, 1.27, 1.71, 2.03])
    loss_exp = 100 - np.array([71.4, 65.7, 54.2, 42.8, 31.4, 19.9, 12.5, 8.5])
    
    w_range = np.linspace(0.05, 2.50, 40)
    tau_model = []
    curr = [-0.0001, 0.0002]
    
    prog = st.progress(0)
    for i, w in enumerate(w_range):
        func = lambda x: obj_func(x, w/s_cr, props, theta_deg, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        if ier == 1:
            # คำนวณ Tau ย้อนกลับ
            res = obj_func(sol, w/s_cr, props, theta_deg, geom) # Dummy calling to check logic if needed
            # Re-calculate Tau specifically for plotting (Simplified for brevity)
            # (Logic ในส่วนนี้เพื่อให้โค้ดสั้นลงแต่ทำงานได้จริง)
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
    
    # Plot
    tau_model = np.array(tau_model)
    tau_u = np.nanmax(tau_model)
    deg = (1 - (tau_model/tau_u))*100
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.plot(w_range, deg, 'r-', lw=3, label='Model')
        ax.plot(w_exp, loss_exp, 'ro', markeredgecolor='k', label='Exp Data')
        ax.set_xlabel('Crack Width (mm)'); ax.set_ylabel('Degradation (%)'); ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.success(f"Max Capacity: {tau_u:.2f} MPa")
else:
    st.info("Click button to run")