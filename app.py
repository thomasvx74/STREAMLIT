import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import fsolve
import cantera as ct
from CoolProp.CoolProp import PropsSI
import os

# =============================================================================
# 1. CONFIGURATION ET UTILITAIRES
# =============================================================================
st.set_page_config(
    page_title="Simulateur Régénératif Avancé",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_patch_spines_invisible(ax):
    """Permet de masquer le cadre des axes décalés pour la lisibilité"""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

st.title("🚀 Simulation Avancée : Tuyère & Refroidissement Régénératif")
st.markdown("""
Ce simulateur modélise les échanges thermiques couplés (Gaz-Paroi-Liquide) dans une tuyère de fusée.
Il intègre l'effet des canaux hélicoïdaux (en spirale) pour augmenter l'efficacité du refroidissement.
""")

# =============================================================================
# 2. BARRE LATÉRALE (PARAMÈTRES D'ENTRÉE)
# =============================================================================
st.sidebar.header("1. Géométrie des Canaux")
N_channels = st.sidebar.number_input("Nombre de canaux", value=100, step=10)
channel_width_mm = st.sidebar.number_input("Largeur canal (mm)", value=1.5, step=0.1)
channel_height_mm = st.sidebar.number_input("Hauteur canal (mm)", value=4.0, step=0.1)
wall_thickness_mm = st.sidebar.number_input("Épaisseur paroi (mm)", value=1.0, step=0.1)
channel_angle = st.sidebar.slider("Angle d'hélice (°)", 0.0, 60.0, 0.0, step=5.0, 
                                  help="0° = canaux droits. Un angle augmente la surface d'échange.")

# Conversion SI
channel_width = channel_width_mm / 1000.0
channel_height = channel_height_mm / 1000.0
wall_thickness = wall_thickness_mm / 1000.0

st.sidebar.header("2. Conditions Moteur")
P_cc_bar = st.sidebar.number_input("Pression Chambre (bar)", value=70.0, step=5.0)
P_cc = P_cc_bar * 1e5
mdot_coolant = st.sidebar.number_input("Débit liquide (kg/s)", value=2.2, step=0.1)
T_coolant_in = st.sidebar.number_input("Temp. Entrée Liquide (K)", value=35.0, step=1.0)
P_coolant_in_bar = st.sidebar.number_input("Pression Liquide (bar)", value=110.0, step=5.0)
P_coolant_in = P_coolant_in_bar * 1e5

st.sidebar.header("3. Matériau")
k_material = st.sidebar.number_input("Conductivité (W/m.K)", value=390.0, 
                                     help="390 pour le Cuivre, 20 pour l'Inconel")

# =============================================================================
# 3. MOTEUR PHYSIQUE (FONCTIONS)
# =============================================================================

@st.cache_data
def generate_geometry():
    """Charge ou génère la géométrie."""
    csv_file = "profil_tuyere.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # On s'assure de renvoyer des tableaux numpy
        return df['x'].values, df['radius'].values
    else:
        x = np.linspace(-0.05, 0.25, 300)
        # Formule sécurisée avec np.abs
        r = np.where(x < 0, 0.03 + x**2 * 8, 0.03 + 0.12 * np.sqrt(np.abs(x)))
        pd.DataFrame({'x': x, 'radius': r}).to_csv(csv_file, index=False)
        return x, r

def get_mach(area_ratio, gamma, subsonic=True):
    def eq(M):
        if M <= 0: return 1e6
        return (1/M)*((2/(gamma+1))*(1+(gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - area_ratio
    guess = 0.1 if subsonic else 2.5
    return fsolve(eq, guess)[0]

def bartz_hg(D_t, P_cc, c_star, At_A, T_cc, T_w, gamma, M, mu_g, cp_g, Pr_g):
    sigma = ((0.5 * T_w/T_cc * (1 + (gamma-1)/2 * M**2) + 0.5)**(-0.68) * (1 + (gamma-1)/2 * M**2)**(-0.12))
    hg = (0.026 / (D_t**0.2)) * (mu_g**0.2 * cp_g / Pr_g**0.6) * (P_cc/c_star)**0.8 * (At_A)**0.9 * sigma
    return hg

def run_simulation_logic():
    """Exécute toute la boucle de calcul physique."""
    
    # 1. Géométrie
    x_coords, r_coords = generate_geometry()
    dr_dx = np.gradient(r_coords, x_coords)
    A = np.pi * r_coords**2
    idx_t = np.argmin(A)
    A_t, D_t = A[idx_t], 2*r_coords[idx_t]

    # 2. Thermodynamique Gaz (Cantera)
    try:
        gas = ct.Solution('gri30.yaml')
        gas.TPX = 300, P_cc, {'H2': 1, 'O2': 0.5} 
        gas.equilibrate('HP')
        T_cc, gamma = gas.T, gas.cp/gas.cv
        mu_g, cp_g, Pr_g = gas.viscosity, gas.cp, 0.7
        R_spec = ct.gas_constant / gas.mean_molecular_weight
        c_star = P_cc * A_t * np.sqrt(gamma)
    except Exception as e:
        st.error(f"Erreur Cantera (gri30.yaml manquant ?): {e}")
        return None

    # 3. Initialisation Tableaux
    N_pts = len(x_coords)
    res = {
        'x': x_coords, 'r': r_coords, 'dr_dx': dr_dx, 'idx_t': idx_t,
        'Vel': np.zeros(N_pts), 'Mach': np.zeros(N_pts),
        'P_gas': np.zeros(N_pts), 'P_cool': np.zeros(N_pts),
        'T_gas': np.zeros(N_pts),
        'T_wh': np.zeros(N_pts), 'T_wc': np.zeros(N_pts),
        'T_cool': np.zeros(N_pts)
    }

    # 4. Boucle de Calcul (Contre-Courant : Sortie -> Entrée)
    curr_Tc = T_coolant_in
    curr_Pc = P_coolant_in
    
    # Facteur d'angle
    angle_rad = np.radians(channel_angle)
    angle_factor = 1.0 / np.cos(angle_rad)

    progress_bar = st.progress(0)

    for i in range(N_pts-1, -1, -1):
        # Aerodynamique 1D
        is_sub = (i < idx_t)
        M = get_mach(A[i]/A_t, gamma, subsonic=is_sub)
        res['Mach'][i] = M
        
        # Pression et Température statiques gaz
        T_st = T_cc / (1 + (gamma-1)/2 * M**2)
        P_st = P_cc / (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))
        res['P_gas'][i] = P_st
        
        T_aw = T_st * (1 + 0.9 * (gamma-1)/2 * M**2)
        res['Vel'][i] = M * np.sqrt(gamma * R_spec * T_st)

        # Propriétés Liquide (CoolProp)
        try:
            rho = PropsSI('D','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            mu  = PropsSI('V','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            cp  = PropsSI('C','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            k_l = PropsSI('L','P',curr_Pc,'T',curr_Tc,'Hydrogen')
        except:
            rho, mu, cp, k_l = 70, 1e-5, 14000, 0.1

        # Coefficients d'échange
        W, H = channel_width, channel_height
        Dh = 4*W*H / (2*(W+H))
        
        Re = (mdot_coolant/(N_channels*W*H)) * Dh / mu
        h_c = 0.023 * (k_l/Dh) * Re**0.8 * (cp*mu/k_l)**0.4
        
        # Ailettes
        t_fin = (2*np.pi*r_coords[i] - N_channels*W) / N_channels
        if t_fin < 1e-5: t_fin = 1e-5
        m_fin = np.sqrt(2 * h_c / (k_material * t_fin))
        eta_f = np.tanh(m_fin * H) / (m_fin * H)
        h_c_eff = h_c * (W + 2 * H * eta_f) / (W + t_fin)
        
        # Gaz (Bartz)
        h_g = bartz_hg(D_t, P_cc, c_star, A_t/A[i], T_cc, 800, gamma, M, mu_g, cp_g, Pr_g)
        
        # Bilan Thermique
        R_tot = (1/h_g) + (wall_thickness/k_material) + (1/(h_c_eff * angle_factor))
        q = (T_aw - curr_Tc) / R_tot
        
        res['T_gas'][i] = T_aw
        res['T_wh'][i]  = T_aw - q/h_g
        res['T_wc'][i]  = res['T_wh'][i] - q*(wall_thickness/k_material)
        res['T_cool'][i]= curr_Tc
        res['P_cool'][i]= curr_Pc
        
        # Avancée Fluide (Mise à jour pour le pas précédent)
        dx = abs(x_coords[i] - x_coords[i-1]) if i > 0 else 0.001
        dx_eff = dx * angle_factor
        
        # 1. Énergie
        curr_Tc += q * (2*np.pi*r_coords[i]*dx) / (mdot_coolant * cp)
        
        # 2. Perte de charge
        # On approxime la vitesse liquide : mdot = rho * V * Area
        v_liq = mdot_coolant / (rho * N_channels * W * H)
        dP = 0.02 * (dx_eff/Dh) * (rho * v_liq**2)/2 # Darcy approx f=0.02
        curr_Pc += dP # On remonte le courant, P augmente
        
        if i % 30 == 0:
            progress_bar.progress((N_pts - i) / N_pts)
            
    progress_bar.progress(1.0)
    return res

# =============================================================================
# 4. INTERFACE UTILISATEUR & GRAPHIQUES
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Simulation & Résultats", "📋 Données Brutes", "🎓 Méthodologie"])

if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None

# --- ONGLET 1 : GRAPHIQUES ---
with tab1:
    st.write("### Contrôle de la Simulation")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Cliquez ci-dessous pour lancer le calcul.")
        launch_btn = st.button("▶️ LANCER LA SIMULATION", use_container_width=True, type="primary")

    if launch_btn:
        with st.spinner("Calculs thermodynamiques en cours..."):
            st.session_state['sim_data'] = run_simulation_logic()
    
    if st.session_state['sim_data']:
        data = st.session_state['sim_data']
        x_c, r_c, vel = data['x'], data['r'], data['Vel']
        st.success("Simulation terminée !")
        
        # ---------------------------------------------------------------------
        # GRAPHIQUE 1 : CHAMP DE VITESSE (Existant)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("1. Champ de Vitesse Mach (2D)")
        

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        Y_MAX = max(r_c) * 1.4
        y_vals = np.linspace(-Y_MAX, Y_MAX, 120)
        X_grid, Y_grid = np.meshgrid(x_c, y_vals)
        
        Vel_Mag_2D = np.zeros_like(X_grid)
        Mask_2D = np.ones_like(X_grid, dtype=bool)

        # Correction Dimensionnelle
        v_interp_1d = np.interp(X_grid[0,:], x_c, vel)
        r_interp_1d = np.interp(X_grid[0,:], x_c, r_c)
        v_interp_2d = np.tile(v_interp_1d, (len(y_vals), 1))
        r_interp_2d = np.tile(r_interp_1d, (len(y_vals), 1))
        dist_norm = np.abs(Y_grid) / r_interp_2d
        mask_valid = dist_norm <= 1.0
        Vel_Mag_2D[mask_valid] = v_interp_2d[mask_valid] * (1 - dist_norm[mask_valid]**2)**(1/7)
        Mask_2D[mask_valid] = False
        Vel_Mag_Ma = np.ma.masked_where(Mask_2D, Vel_Mag_2D)
        
        cmap = plt.get_cmap('turbo')
        mesh = ax1.pcolormesh(X_grid, Y_grid, Vel_Mag_Ma, cmap=cmap, shading='auto')
        plt.colorbar(mesh, ax=ax1, label="Vitesse (m/s)")
        
        idx_sections = np.linspace(15, len(x_c)-15, 10, dtype=int)
        for idx in idx_sections:
            xi, ri = x_c[idx], r_c[idx]
            dridx = data['dr_dx'][idx]
            v_center = vel[idx]
            y_vecs = np.linspace(-ri*0.9, ri*0.9, 8)
            dist_n = np.abs(y_vecs) / ri
            v_mag = v_center * (1 - dist_n**2)**(1/7)
            theta_flow = np.arctan(dridx * (y_vecs/ri))
            ax1.quiver(np.full_like(y_vecs, xi), y_vecs, v_mag * np.cos(theta_flow), v_mag * np.sin(theta_flow), 
                       color='white', edgecolor='black', linewidth=0.5, width=0.003, headwidth=3)

        ax1.plot(x_c, r_c, 'k', linewidth=2)
        ax1.plot(x_c, -r_c, 'k', linewidth=2)
        ax1.set_ylim(-Y_MAX, Y_MAX)
        ax1.set_xlabel("Position Axiale (m)")
        ax1.set_ylabel("Rayon (m)")
        st.pyplot(fig1)

        # ---------------------------------------------------------------------
        # GRAPHIQUE 2 : PROFILS THERMIQUES (Existant)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Profils Thermiques aux Interfaces")
        

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = [20, data['idx_t'], len(x_c)-20]
        labels = ["Entrée", "Col", "Sortie"]
        colors = ["green", "red", "blue"]
        
        for k, idx in enumerate(indices):
            Ti, Twh, Twc, Tliq = data['T_gas'][idx], data['T_wh'][idx], data['T_wc'][idx], data['T_cool'][idx]
            x_g, y_g = np.linspace(-0.5, 0, 15), Twh + (Ti - Twh)*(1 - np.exp(np.linspace(-0.5, 0, 15)/0.1))
            x_w, y_w = np.linspace(0, 1, 10), np.linspace(Twh, Twc, 10)
            x_l, y_l = np.linspace(1, 1.5, 15), Tliq + (Twc - Tliq)*np.exp(-(np.linspace(1, 1.5, 15)-1)/0.1)
            
            full_x, full_y = np.concatenate([x_g, x_w, x_l]), np.concatenate([y_g, y_w, y_l])
            ax2.plot(full_x, full_y, color=colors[k], linewidth=2, label=labels[k])
            
            bbox = dict(boxstyle="round,pad=0.3", fc="white", ec=colors[k], alpha=0.8)
            ax2.text(0, Twh + (60 if k==1 else -60), f"{int(Twh)} K", fontsize=9, color=colors[k], fontweight='bold', ha='right', bbox=bbox)
            ax2.text(1, Twc, f"{int(Twc)} K", fontsize=9, color=colors[k], fontweight='bold', ha='left', bbox=bbox)

        ax2.axvspan(0, 1, color='orange', alpha=0.1, label='Paroi')
        ax2.set_xlabel("Zone Normalisée (Gaz -> Paroi -> Liquide)")
        ax2.set_ylabel("Température (K)")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig2)

        # ---------------------------------------------------------------------
        # GRAPHIQUE 3 : VUE GLOBALE MULTI-ÉCHELLES (NOUVEAU !)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("3. Vue d'Ensemble : Températures et Mach")
        
        fig3, host = plt.subplots(figsize=(12, 6))
        fig3.subplots_adjust(right=0.75) # Espace à droite pour les axes

        par1 = host.twinx()
        par2 = host.twinx()
        par3 = host.twinx()

        # Décalage des axes
        par2.spines["right"].set_position(("axes", 1.08))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        par3.spines["right"].set_position(("axes", 1.16))
        make_patch_spines_invisible(par3)
        par3.spines["right"].set_visible(True)

        # Tracé
        p_geo, = host.plot(x_c, r_c*100, "k-", linewidth=3, alpha=0.4, label="Géométrie")
        host.plot(x_c, -r_c*100, "k-", linewidth=3, alpha=0.4)
        host.fill_between(x_c, r_c*100, -r_c*100, color='gray', alpha=0.1)

        p_gas, = par1.plot(x_c, data['T_gas'], "r-", label="T° Gaz")
        p_wall, = par1.plot(x_c, data['T_wh'], "orange", linestyle="--", label="T° Paroi (Chaude)")
        p_cool, = par2.plot(x_c, data['T_cool'], "b-", linewidth=2, label="T° Liquide")
        p_mach, = par3.plot(x_c, data['Mach'], "g-.", label="Mach")

        # Labels
        host.set_xlabel("Position (m)")
        host.set_ylabel("Rayon (cm)")
        
        par1.set_ylabel("Température Haute (K)")
        par1.yaxis.label.set_color(p_gas.get_color())
        
        par2.set_ylabel("Température Liquide (K)")
        par2.yaxis.label.set_color(p_cool.get_color())
        
        par3.set_ylabel("Nombre de Mach")
        par3.yaxis.label.set_color(p_mach.get_color())

        lines = [p_geo, p_gas, p_wall, p_cool, p_mach]
        host.legend(lines, [l.get_label() for l in lines], loc='upper left')
        host.grid(True, alpha=0.3)
        
        st.pyplot(fig3)

        # ---------------------------------------------------------------------
        # GRAPHIQUE 4 : PRESSIONS (NOUVEAU !)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("4. Évolution des Pressions")
        
        fig4, ax_main = plt.subplots(figsize=(10, 5))
        ax_pg = ax_main.twinx() # Axe Pression Gaz
        ax_pc = ax_main.twinx() # Axe Pression Coolant
        
        # Décalage pour éviter chevauchement
        ax_pc.spines["right"].set_position(("axes", 1.12))
        make_patch_spines_invisible(ax_pc)
        ax_pc.spines["right"].set_visible(True)

        # Géométrie en fond
        ax_main.fill_between(x_c, r_c*100, (r_c+0.005)*100, color='gray', alpha=0.3, label="Paroi")
        ax_main.fill_between(x_c, -r_c*100, -(r_c+0.005)*100, color='gray', alpha=0.3)
        
        l_pg, = ax_pg.plot(x_c, data['P_gas']/1e5, "r-.", linewidth=2, label="Pression Gaz")
        l_pc, = ax_pc.plot(x_c, data['P_cool']/1e5, "b-", linewidth=2, label="Pression H2")

        ax_main.set_ylabel("Rayon (cm)", color='gray')
        ax_main.set_xlabel("Position Axiale (m)")
        
        ax_pg.set_ylabel("Pression Gaz (bar)", color='r')
        ax_pg.tick_params(axis='y', colors='r')
        
        ax_pc.set_ylabel("Pression Liquide (bar)", color='b')
        ax_pc.tick_params(axis='y', colors='b')
        
        lns = [l_pg, l_pc]
        ax_main.legend(lns, [l.get_label() for l in lns], loc='upper right')
        ax_main.grid(True, alpha=0.3)
        
        st.pyplot(fig4)

        # ---------------------------------------------------------------------
        # GRAPHIQUE 5 : COUPE RADIALE ZOOMÉE (NOUVEAU !)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("5. Zoom : Coupe Radiale dans le Divergent")
        
        # On cible x = 2 * diamètre au col
        D_t = 2 * min(r_c)
        target_x = 2 * D_t
        idx_target = (np.abs(x_c - target_x)).argmin()
        
        vals = {k: data[k][idx_target] for k in ['T_gas', 'T_wh', 'T_wc', 'T_cool']}
        
        # Génération profils micro
        dist_g = np.linspace(-0.5e-3, 0, 20) 
        prof_g = vals['T_wh'] + (vals['T_gas'] - vals['T_wh']) * (1 - np.exp((dist_g)/0.0001))
        
        dist_w = np.linspace(0, wall_thickness, 10)
        prof_w = np.linspace(vals['T_wh'], vals['T_wc'], 10)
        
        dist_c = np.linspace(wall_thickness, wall_thickness + 0.5e-3, 20)
        prof_c = vals['T_cool'] + (vals['T_wc'] - vals['T_cool']) * np.exp(-(dist_c - wall_thickness)/0.0001)

        fig5, ax5 = plt.subplots(figsize=(10, 5))
        ax5.plot(dist_g*1000, prof_g, 'r--', label='Couche Limite Gaz')
        ax5.plot(dist_w*1000, prof_w, 'k-', linewidth=4, label='Paroi Cuivre')
        ax5.plot(dist_c*1000, prof_c, 'b--', label='Couche Limite Liquide')
        
        ax5.plot(0, vals['T_wh'], 'ro')
        ax5.plot(wall_thickness*1000, vals['T_wc'], 'bo')
        ax5.axvspan(0, wall_thickness*1000, color='orange', alpha=0.2)
        
        # Annotations
        ax5.text(-0.4, vals['T_gas']-200, "GAZ", color='red', fontweight='bold')
        ax5.text(wall_thickness*1000/2, (vals['T_wh']+vals['T_wc'])/2, "PAROI", ha='center', fontweight='bold', rotation=90)
        ax5.text(wall_thickness*1000 + 0.1, vals['T_cool']+20, "H2 LIQUIDE", color='blue', fontweight='bold')

        ax5.set_xlabel("Distance (mm) - 0 = Interface Paroi/Gaz")
        ax5.set_ylabel("Température (K)")
        ax5.set_title(f"Profil Radial à x={x_c[idx_target]:.3f} m")
        ax5.grid(True, linestyle=':')
        ax5.legend()
        
        st.pyplot(fig5)

# --- ONGLET 2 : DONNÉES BRUTES (NOUVEAU) ---
with tab2:
    st.header("📋 Tableau de Données Brutes")
    
    if st.session_state['sim_data']:
        data = st.session_state['sim_data']
        
        # Création du DataFrame
        df_results = pd.DataFrame({
            "Position X (m)": data['x'],
            "Rayon (m)": data['r'],
            "Mach": data['Mach'],
            "Vitesse Gaz (m/s)": data['Vel'],
            "Temp. Gaz (K)": data['T_gas'],
            "Temp. Paroi Chaude (K)": data['T_wh'],
            "Temp. Paroi Froide (K)": data['T_wc'],
            "Temp. Liquide (K)": data['T_cool']
        })
        
        # Affichage interactif
        st.dataframe(df_results, use_container_width=True)
        
        # Bouton de téléchargement
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données en CSV",
            data=csv,
            file_name='resultats_simulation_tuyere.csv',
            mime='text/csv',
        )
    else:
        st.warning("⚠️ Veuillez lancer la simulation dans le premier onglet pour afficher les données.")

# --- ONGLET 3 : MÉTHODOLOGIE & EXPLICATIONS ---

    # --- ONGLET 3 : DOCUMENTATION & MÉTHODOLOGIE ---
with tab3:
    st.markdown("# 📘 Architecture & Méthodologie")
    
    st.info("""
    **Résumé :** Ce simulateur est un code **Quasi-1D Stationnaire**. Il découpe la tuyère en fines tranches et calcule l'équilibre thermique sur chacune d'elles, en prenant en compte la variation de section.
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 1. ARCHITECTURE LOGICIELLE
    # -------------------------------------------------------------------------
    st.header("1. Les Outils (Librairies)")
    st.write("Le code s'appuie sur des standards de l'industrie pour garantir la précision :")

    c1, c2 = st.columns(2)
    with c1:
        st.success("**🧪 Cantera (Chimie)**")
        st.caption("Calcul de Combustion")
        st.write("Au lieu de fixer des constantes, Cantera calcule l'équilibre chimique réel H2/O2 à haute température. Il nous donne le **Gamma** ($\gamma$) et la **Température de flamme** exacts.")

    with c2:
        st.info("**❄️ CoolProp (Fluides)**")
        st.caption("Propriétés H2 Liquide")
        st.write("L'hydrogène change radicalement de comportement selon la pression. CoolProp fournit la densité, viscosité et conductivité locales précises à chaque millimètre.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. ALGORITHME VISUEL
    # -------------------------------------------------------------------------
    st.header("2. L'Algorithme de Résolution")
    st.write("Le calcul suit la logique physique du fluide de refroidissement : il remonte le courant.")

    st.graphviz_chart('''
    digraph {
        rankdir=TB;
        node [fontname="Arial", fontsize=11, style="filled,rounded", penwidth=0];
        edge [color="#a0a0a0", penwidth=1.5];

        // INITIALISATION
        subgraph cluster_init {
            label = ""; penwidth=0;
            Start [label="🚀 DÉPART\nGéométrie & Paramètres", shape=parallelogram, fillcolor="#2196F3", fontcolor="white"];
            Chem [label="🔥 COMBUSTION\n(Cantera)\nCalcul T_flamme, Gamma", shape=cylinder, fillcolor="#FFCCBC"];
        }

        // BOUCLE
        subgraph cluster_loop {
            label = " BOUCLE PRINCIPALE (On remonte de la Sortie vers l'Injecteur) ";
            style=filled; bgcolor="#F5F5F5"; color="#dddddd";
            fontcolor="#555555";
            
            node [shape=box];
            
            // Étape 1 : Gaz
            Gas [label="1. AÉRODYNAMIQUE GAZ\nMach = f(Section)\nT_gaz, P_gaz", fillcolor="#FFAB91"];
            
            // Étape 2 : Liquide
            Liq [label="2. ÉTAT LIQUIDE\n(CoolProp)\nDensité, Viscosité...", fillcolor="#81D4FA"];
            
            // Étape 3 : Échange
            Exchange [label="3. ÉCHANGES THERMIQUES\nh_gaz (Bartz) ⚡ h_liq (Dittus)", fillcolor="#FFF59D"];
            
            // Étape 4 : Bilan
            Bilan [label="4. BILAN DE FLUX\nq = Delta T / Résistances", shape=diamond, fillcolor="#A5D6A7"];
            
            // Étape 5 : Mise à jour
            Update [label="5. MISE À JOUR\nT_liq += Énergie Gagnée\nP_liq += Frottements", fillcolor="#81D4FA"];
        }

        End [label="🏁 RÉSULTATS\nAffichage Graphiques", shape=parallelogram, fillcolor="#4CAF50", fontcolor="white"];

        // FLUX
        Start -> Chem -> Gas;
        Gas -> Liq -> Exchange -> Bilan -> Update;
        Update -> Gas [label=" Tranche suivante (i-1)", color="#1976D2", style=dashed];
        Update -> End [label=" i=0 (Fini)"];
    }
    ''')

    st.markdown("---") 

    # -------------------------------------------------------------------------
    # 3. PHYSIQUE SIMPLIFIÉE
    # -------------------------------------------------------------------------
    st.header("3. Fonctionnement Physique (Quasi-1D)")

    st.markdown("""
    **Pourquoi "Quasi-1D" ?**
    Au lieu de simuler tout le volume en 3D (ce qui prendrait des heures), nous supposons que les propriétés (Pression, Température, Vitesse) sont uniformes sur une coupe transversale (une tranche).
    """)
    

    col_phys1, col_phys2 = st.columns([1, 1.2])

    with col_phys1:
        st.subheader("⚡ Le Circuit Thermique")
        st.write("Le flux de chaleur $q$ doit traverser 3 obstacles (résistances) :")
        st.markdown("""
        1.  **La couche limite Gaz** (Résistance convective)
        2.  **La paroi en Cuivre** (Résistance conductive)
        3.  **La couche limite Liquide** (Résistance convective)
        """)
        
    with col_phys2:
        st.subheader("🧮 L'Équation Bilan")
        st.info("Le calcul repose sur cette unique formule d'équilibre :")
        st.latex(r"""
        q = \frac{T_{gaz} - T_{liquide}}{ \underbrace{\frac{1}{h_g}}_{\text{Gaz}} + \underbrace{\frac{e}{k}}_{\text{Paroi}} + \underbrace{\frac{1}{h_{liq} \cdot \Omega}}_{\text{Liquide}} }
        """)
        st.caption("Où $\Omega$ représente l'augmentation de surface due aux **ailettes** et à **l'angle** des canaux.")

    st.markdown("---")
