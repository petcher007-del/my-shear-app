import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings
from PIL import Image
import matplotlib.cm as cm

# ปิดข้อความแจ้งเตือน
warnings.filterwarnings("ignore")

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="General Shear Assessment", layout="wide")

st.title("🏗️ Structural Shear Strength Assessment")
st.markdown("""
**Method:** Sigma-x Analysis (Analytical Model)  
*Analysis: Shear Strength vs. Crack Width (with Angle Sensitivity).*
""")

# ==========================================
# 1. INPUTS
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    # --- 1. Material ---
    with st.expander("1. Material Properties", expanded=False):
        fc_prime = st.number_input("f'c (MPa)", value=31.5, step=0.5)
        default_Ec = 4700 * np.sqrt(fc_prime)
        Es = st.number_input("Es (MPa)", value=200000.0)
        Ec = default_Ec 

    # --- 2. Reinforcement ---
    with st.expander("2. Reinforcement", expanded=False):
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
        st.markdown("### A. Main Angle")
        theta_deg = st.number_input("Crack Angle (deg)", value=46.0)
        
        # === NEW FEATURE: Multi-Angle Comparison ===
        st.markdown("---")
        st.markdown("### 🔍 Sensitivity Analysis")
        compare_angles = st.checkbox("Compare Multiple Angles?", value=True)
        if compare_angles:
            delta_deg = st.slider("Angle Variation (± deg)", 5, 20, 10, help="Plot lines for Main Angle ± this value")
        
        st.markdown("---")
        st.markdown("### B. Crack Spacing ($s_{cr}$)")
        scr_method = st.radio("Method:", ["Manual Input", "Calculate from Rebar"], horizontal=True, label_visibility="collapsed")
        
        if scr_method == "Manual Input":
            s_cr = st.number_input("Crack Spacing (mm)", value=268.0)
        else:
            col_rebar1, col_rebar2 = st.columns(2)
            s_mx = col_rebar1.number_input("Sx (mm)", value=300.0)
            s_my = col_rebar2.number_input("Sy (mm)", value=200.0)
            th_rad = np.deg2rad(theta_deg)
            s_cr_calc = 1 / ((np.abs(np.sin(th_rad))/(s_mx+1e-9)) + (np.abs(np.cos(th_rad))/(s_my+1e-9)))
            st.caption(f"Calculated $s_{{cr}}$ (for {theta_deg}°): **{s_cr_calc:.1f} mm**")
            s_cr = s_cr_calc

        st.markdown("---")
        st.markdown("### C. Analysis Range")
        w_end = st.number_input("Max Width (mm)", value=2.5)
        w_step = st.number_input("Step Size (mm)", value=0.05)

    # --- 4. Boundary & Data ---
    with st.expander("4. Boundary & Experimental Data", expanded=False):
        use_clamping = st.checkbox("Apply Clamping Forces?", value=True)
        if use_clamping:
            h_av, av = 1067.0, 1767.0
            x_cr1, x_cr2 = 681.0, 655.0
        else:
            h_av, av, x_cr1, x_cr2 = 0, 1, 0, 0
            
        default_data = pd.DataFrame([{"Width": 0.05, "Loss": 28.6}, {"Width": 0.79, "Loss": 57.2}, {"Width": 2.03, "Loss": 91.5}])
        edited_df = st.data_editor(default_data, num_rows="dynamic", hide_index=True)

# Pack variables
props = {'fc_prime': fc_prime, 'Ec': Ec, 'Es': Es, 'rho_l': rho_l, 'rho_v': rho_v, 'rho_h': rho_h, 'fy_l': fy_l, 'fy_v': fy_v, 'fy_h': fy_h}
geom = {'h_av': h_av, 'av': av, 'x_cr1': x_cr1, 'x_cr2': x_cr2, 'use_clamping': use_clamping}

# ==========================================
# 2. SOLVER FUNCTION
# ==========================================
def obj_func(x, eps_1, props, theta_val, geom): # Added theta_val as argument
    eps_2, gam_cr = x[0], x[1]
    th = np.deg2rad(theta_val)
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

def run_simulation(angle_in, w_array, spacing):
    res_tau = []
    curr = [-0.0001, 0.0002]
    for w in w_array:
        # Pass the specific angle 'angle_in' to the objective function
        func = lambda x: obj_func(x, w/spacing, props, angle_in, geom)
        sol, _, ier, _ = fsolve(func, curr, full_output=True)
        if ier == 1:
            # Re-calculate Tau for result
            # (Repeating logic for display)
            th=np.deg2rad(angle_in); s,c=np.sin(th),np.cos(th); s2,c2,sc=s**2,c**2,s*c
            fc1 = (0.33*np.sqrt(props['fc_prime']))/(1+np.sqrt(633*(w/spacing))); fc1=min(fc1,4.2)
            term=(-(w/spacing)/(sol[0] if sol[0]!=0 else 1e-9))-0.28
            bd=1.0 if term<0 else 1/(1+0.27*(term**0.8))
            if bd>1: bd=1.0
            r=sol[0]/(bd*-0.002); fc2=0 if r<0 else -bd*props['fc_prime']*(2*r-r**2)
            vci=3.83*(props['fc_prime']**(1/3))*(sol[1]**2/((w/spacing)**2+sol[1]**2))
            tau=(fc1-fc2)*sc+vci*(s2-c2)
            res_tau.append(tau)
            curr = sol
        else:
            res_tau.append(np.nan)
    return np.array(res_tau)

# ==========================================
# 3. MAIN APP EXECUTION
# ==========================================
if st.button("🚀 Run Analysis", type="primary"):
    
    w_range = np.arange(0.001, w_end + (w_step/100), w_step)
    
    # 1. Run Main Analysis (Main Angle)
    with st.spinner(f"Analyzing for {theta_deg}°..."):
        tau_main = run_simulation(theta_deg, w_range, s_cr)
    
    # 2. Run Comparison Analyses (if checked)
    results_dict = {f"{theta_deg}° (Main)": tau_main}
    
    if compare_angles:
        ang_low = theta_deg - delta_deg
        ang_high = theta_deg + delta_deg
        
        # Recalculate Scr for different angles if "Calculate" method is used
        # (Because spacing changes with angle in theory, but here we keep user input or recalculate?)
        # For simplicity in "Manual Mode", we keep s_cr constant.
        # For "Calculate Mode", we should technically update s_cr.
        # Let's keep s_cr constant for fair comparison of "Angle Effect Only" unless refined.
        
        with st.spinner(f"Analyzing for {ang_low}° and {ang_high}°..."):
            tau_low = run_simulation(ang_low, w_range, s_cr)
            tau_high = run_simulation(ang_high, w_range, s_cr)
            
        results_dict[f"{ang_low}° (-{delta_deg})"] = tau_low
        results_dict[f"{ang_high}° (+{delta_deg})"] = tau_high

    # Process Main Result for Metrics
    tau_u = np.nanmax(tau_main)
    
    # --- PLOTTING ---
    st.divider()
    col_plot, col_info = st.columns([3, 1])
    
    with col_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
        styles = ['-', '--', '--']
        widths = [2.5, 1.5, 1.5]
        
        # Plot all curves
        for i, (label, tau_vals) in enumerate(results_dict.items()):
            deg_curve = (1 - (tau_vals/np.nanmax(tau_vals)))*100 # Normalize to ITS OWN max? Or global max?
            # Usually we plot Absolute Shear Strength or Degradation relative to the MAIN model.
            # Let's plot ABSOLUTE SHEAR STRENGTH first (Clearer)
            # OR Degradation % relative to the max of that specific angle.
            
            # Let's Plot Shear Strength (MPa) - It's more informative for comparison
            ax.plot(w_range, tau_vals, label=label, color=colors[i%3], linestyle=styles[i%3], linewidth=widths[i%3])

        # Plot User Data (Scaled to Shear Strength if possible, but user data is usually Loss %)
        # If User Data is Loss %, we need a secondary axis or plot Loss % graph.
        # Let's switch to plotting LOSS (%) to match previous context perfectly.
        
        ax.clear() # Reset to plot Loss (%)
        for i, (label, tau_vals) in enumerate(results_dict.items()):
            # Calculate degradation based on THAT curve's max (Internal degradation)
            local_max = np.nanmax(tau_vals)
            degradation = (1 - (tau_vals/local_max))*100
            ax.plot(w_range, degradation, label=f"{label} (Max $\\tau$={local_max:.2f})", 
                   color=colors[i%3], linestyle=styles[i%3], linewidth=widths[i%3])
            
        # Plot Experimental Data points
        if not edited_df.empty:
             ax.plot(edited_df["Width"], edited_df["Loss"], 'ko', markersize=8, label='Exp. Data')

        ax.set_xlabel('Crack Width (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Shear Strength Degradation (%)', fontweight='bold', fontsize=12)
        ax.set_title('Impact of Crack Angle on Shear Degradation', fontsize=14)
        ax.set_xlim(0, w_end)
        ax.set_ylim(0, 100)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
        st.pyplot(fig)
        st.caption(f"💡 Note: Solid Red Line is your main input ({theta_deg}°). Dashed lines are for comparison.")

    with col_info:
        st.subheader("📊 Analysis Results")
        st.metric("Max Strength (Main)", f"{tau_u:.2f} MPa")
        st.markdown(f"**At {theta_deg}°:**")
        st.write(f"- $s_{{cr}}$ used: {s_cr:.1f} mm")
        st.write(f"- Steps analyzed: {len(w_range)}")
        
        if compare_angles:
            st.markdown("---")
            st.markdown("**Comparison:**")
            st.write(f"🔹 {ang_low}° Max: **{np.nanmax(tau_low):.2f} MPa**")
            st.write(f"🟢 {ang_high}° Max: **{np.nanmax(tau_high):.2f} MPa**")

else:
    st.info("👈 Set parameters and click **Run Analysis** to see the comparison.")
