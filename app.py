import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import fsolve
import cantera as ct
from CoolProp.CoolProp import PropsSI
import os
import base64

# ==============================================================================
# CONFIGURATION GÉNÉRALE DE L'APPLICATION
# ==============================================================================
st.set_page_config(
    page_title="Simulateur Régénératif Avancé",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_patch_spines_invisible(ax):
    """
    Utilitaire graphique pour Matplotlib.
    Masque les bordures (spines) des axes secondaires pour les graphiques multi-échelles.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# ------------------------------------------------------------------------------
# EN-TÊTE ET PRÉSENTATION
# ------------------------------------------------------------------------------
st.title("🚀 Simulation Avancée : Tuyère & Refroidissement Régénératif")
st.markdown("""
Ce simulateur modélise les **échanges thermiques couplés** (Gaz de combustion - Paroi - Liquide de refroidissement) 
dans une tuyère de moteur-fusée.

Le modèle résout l'équilibre thermique en régime stationnaire en prenant en compte :
* La chimie complexe des gaz (Cantera).
* Les propriétés réelles du fluide cryogénique (CoolProp).
* L'effet géométrique des canaux hélicoïdaux (augmentation de surface).
""")

# ==============================================================================
# 1. PARAMÉTRAGE (SIDEBAR)
# ==============================================================================
st.sidebar.header("1. Géométrie des Canaux")
# Paramètres définissant la structure interne de l'échangeur de chaleur (la paroi)
N_channels = st.sidebar.number_input("Nombre de canaux", value=100, step=10)
channel_width_mm = st.sidebar.number_input("Largeur canal (mm)", value=1.5, step=0.1)
channel_height_mm = st.sidebar.number_input("Hauteur canal (mm)", value=4.0, step=0.1)
wall_thickness_mm = st.sidebar.number_input("Épaisseur paroi (mm)", value=1.0, step=0.1)
channel_angle = st.sidebar.slider("Angle d'hélice (°)", 0.0, 60.0, 0.0, step=5.0, 
                                  help="0° = canaux droits. Un angle augmente le temps de séjour du fluide et la surface d'échange.")

# Conversion immédiate en unités SI (Mètres) pour les calculs internes
channel_width = channel_width_mm / 1000.0
channel_height = channel_height_mm / 1000.0
wall_thickness = wall_thickness_mm / 1000.0

st.sidebar.header("2. Conditions Moteur")
# Conditions aux limites thermodynamiques
P_cc_bar = st.sidebar.number_input("Pression Chambre (bar)", value=70.0, step=5.0)
P_cc = P_cc_bar * 1e5
mdot_coolant = st.sidebar.number_input("Débit liquide (kg/s)", value=2.2, step=0.1)
T_coolant_in = st.sidebar.number_input("Temp. Entrée Liquide (K)", value=35.0, step=1.0)
P_coolant_in_bar = st.sidebar.number_input("Pression Liquide (bar)", value=110.0, step=5.0)
P_coolant_in = P_coolant_in_bar * 1e5

st.sidebar.header("3. Matériau")
k_material = st.sidebar.number_input("Conductivité (W/m.K)", value=390.0, 
                                     help="Exemples : ~390 pour le Cuivre (très conducteur), ~20 pour l'Inconel (résistant mais isolant).")

# ==============================================================================
# 2. MOTEUR PHYSIQUE (FONCTIONS CŒUR)
# ==============================================================================

@st.cache_data
def generate_geometry():
    """
    Génère ou charge le profil 2D de la tuyère (Rayon vs Position X).
    
    Returns:
        tuple: (x_coords, r_coords) vecteurs numpy des positions et rayons.
    """
    csv_file = "profil_tuyere.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df['x'].values, df['radius'].values
    else:
        # Profil générique "Rao" simplifié si aucun fichier n'est présent
        x = np.linspace(-0.05, 0.25, 300)
        # Utilisation de np.where pour définir la forme convergente/divergente
        r = np.where(x < 0, 0.03 + x**2 * 8, 0.03 + 0.12 * np.sqrt(np.abs(x)))
        pd.DataFrame({'x': x, 'radius': r}).to_csv(csv_file, index=False)
        return x, r

def get_mach(area_ratio, gamma, subsonic=True):
    """
    Résout l'équation aire-Mach pour un écoulement isentropique compressible.
    
    Args:
        area_ratio (float): Rapport A/A_col.
        gamma (float): Coefficient de Laplace (Cp/Cv).
        subsonic (bool): Si True, cherche la solution M < 1, sinon M > 1.
    
    Returns:
        float: Le nombre de Mach local.
    """
    def eq(M):
        if M <= 0: return 1e6 # Évite les erreurs mathématiques
        return (1/M)*((2/(gamma+1))*(1+(gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - area_ratio
    
    guess = 0.1 if subsonic else 2.5
    return fsolve(eq, guess)[0]

def bartz_hg(D_t, P_cc, c_star, At_A, T_cc, T_w, gamma, M, mu_g, cp_g, Pr_g):
    """
    Calcule le coefficient de convection gaz (h_g) via la corrélation de Bartz.
    Cette corrélation semi-empirique est standard pour estimer les flux thermiques dans les fusées.
    """
    # Facteur de correction sigma pour les effets de couche limite compressible
    sigma = ((0.5 * T_w/T_cc * (1 + (gamma-1)/2 * M**2) + 0.5)**(-0.68) * (1 + (gamma-1)/2 * M**2)**(-0.12))
    
    # Formule de Bartz simplifiée
    hg = (0.026 / (D_t**0.2)) * (mu_g**0.2 * cp_g / Pr_g**0.6) * (P_cc/c_star)**0.8 * (At_A)**0.9 * sigma
    return hg

def run_simulation_logic():
    """
    Orchestre la simulation complète :
    1. Initialisation géométrique.
    2. Calcul de l'équilibre chimique (Cantera).
    3. Boucle itérative spatiale (Contre-courant).
    4. Résolution du bilan thermique à chaque pas.
    """
    
    # --- A. PRÉPARATION GÉOMÉTRIQUE ---
    x_coords, r_coords = generate_geometry()
    dr_dx = np.gradient(r_coords, x_coords) # Pente locale pour l'angle du flux
    A = np.pi * r_coords**2
    idx_t = np.argmin(A) # Index du col (throat)
    A_t, D_t = A[idx_t], 2*r_coords[idx_t]

    # --- B. THERMODYNAMIQUE (COMBUSTION) ---
    try:
        # Initialisation de Cantera pour un mélange H2/O2
        gas = ct.Solution('gri30.yaml')
        gas.TPX = 300, P_cc, {'H2': 1, 'O2': 0.5} 
        gas.equilibrate('HP') # Combustion à Enthalpie et Pression constantes
        
        # Extraction des propriétés du gaz brûlé
        T_cc, gamma = gas.T, gas.cp/gas.cv
        mu_g, cp_g, Pr_g = gas.viscosity, gas.cp, 0.7
        R_spec = ct.gas_constant / gas.mean_molecular_weight
        c_star = P_cc * A_t * np.sqrt(gamma) # Vitesse caractéristique
    except Exception as e:
        st.error(f"Erreur d'initialisation Cantera (vérifiez gri30.yaml) : {e}")
        return None

    # --- C. INITIALISATION DES VECTEURS DE RÉSULTATS ---
    N_pts = len(x_coords)
    res = {
        'x': x_coords, 'r': r_coords, 'dr_dx': dr_dx, 'idx_t': idx_t,
        'Vel': np.zeros(N_pts), 'Mach': np.zeros(N_pts),
        'P_gas': np.zeros(N_pts), 'P_cool': np.zeros(N_pts),
        'T_gas': np.zeros(N_pts),
        'T_wh': np.zeros(N_pts), # Température paroi côté chaud (hot)
        'T_wc': np.zeros(N_pts), # Température paroi côté froid (cold)
        'T_cool': np.zeros(N_pts)
    }

    # --- D. BOUCLE DE RÉSOLUTION (CONTRE-COURANT) ---
    # On parcourt la tuyère de la SORTIE vers l'ENTRÉE (index inversé)
    # car le liquide de refroidissement remonte la tuyère.
    curr_Tc = T_coolant_in
    curr_Pc = P_coolant_in
    
    # Facteur géométrique lié à l'hélicoïde (le chemin est plus long que la tuyère)
    angle_rad = np.radians(channel_angle)
    angle_factor = 1.0 / np.cos(angle_rad)

    progress_bar = st.progress(0)

    for i in range(N_pts-1, -1, -1):
        # 1. Aérodynamique Gaz (Isentropique 1D)
        is_sub = (i < idx_t) # Subsonique avant le col
        M = get_mach(A[i]/A_t, gamma, subsonic=is_sub)
        res['Mach'][i] = M
        
        # Température et Pression statiques (locales)
        T_st = T_cc / (1 + (gamma-1)/2 * M**2)
        P_st = P_cc / (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))
        res['P_gas'][i] = P_st
        
        # Température adiabatique de paroi (Recovery Temperature)
        T_aw = T_st * (1 + 0.9 * (gamma-1)/2 * M**2)
        res['Vel'][i] = M * np.sqrt(gamma * R_spec * T_st)

        # 2. Propriétés du Liquide (CoolProp - Fluide Réel)
        try:
            # Appel à CoolProp pour densité, viscosité, Cp, conductivité
            rho = PropsSI('D','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            mu  = PropsSI('V','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            cp  = PropsSI('C','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            k_l = PropsSI('L','P',curr_Pc,'T',curr_Tc,'Hydrogen')
        except:
            # Valeurs de repli (fallback) en cas d'erreur CoolProp
            rho, mu, cp, k_l = 70, 1e-5, 14000, 0.1

        # 3. Coefficients d'échange thermique
        W, H = channel_width, channel_height
        Dh = 4*W*H / (2*(W+H)) # Diamètre hydraulique
        
        # Nombre de Reynolds et Nusselt (Dittus-Boelter)
        Re = (mdot_coolant/(N_channels*W*H)) * Dh / mu
        h_c = 0.023 * (k_l/Dh) * Re**0.8 * (cp*mu/k_l)**0.4
        
        # Efficacité des ailettes (Fin Efficiency)
        # La chaleur doit voyager le long des "murs" des canaux
        t_fin = (2*np.pi*r_coords[i] - N_channels*W) / N_channels
        if t_fin < 1e-5: t_fin = 1e-5
        m_fin = np.sqrt(2 * h_c / (k_material * t_fin))
        eta_f = np.tanh(m_fin * H) / (m_fin * H)
        h_c_eff = h_c * (W + 2 * H * eta_f) / (W + t_fin) # Coefficient effectif corrigé
        
        # Coefficient Gaz (Bartz)
        h_g = bartz_hg(D_t, P_cc, c_star, A_t/A[i], T_cc, 800, gamma, M, mu_g, cp_g, Pr_g)
        
        # 4. Bilan Thermique (Réseau de résistances)
        # R_tot = R_conv_gaz + R_cond_paroi + R_conv_liq
        R_tot = (1/h_g) + (wall_thickness/k_material) + (1/(h_c_eff * angle_factor))
        q = (T_aw - curr_Tc) / R_tot # Flux thermique (W/m2)
        
        # Calcul des températures intermédiaires
        res['T_gas'][i] = T_aw
        res['T_wh'][i]  = T_aw - q/h_g                  # T° Paroi côté chaud
        res['T_wc'][i]  = res['T_wh'][i] - q*(wall_thickness/k_material) # T° Paroi côté froid
        res['T_cool'][i]= curr_Tc
        res['P_cool'][i]= curr_Pc
        
        # 5. Mise à jour de l'état du fluide pour le pas suivant (précédent en x)
        dx = abs(x_coords[i] - x_coords[i-1]) if i > 0 else 0.001
        dx_eff = dx * angle_factor # Longueur réelle parcourue dans l'hélice
        
        # Échauffement du fluide (Conservation de l'énergie)
        curr_Tc += q * (2*np.pi*r_coords[i]*dx) / (mdot_coolant * cp)
        
        # Perte de charge (Darcy-Weisbach approximé)
        v_liq = mdot_coolant / (rho * N_channels * W * H)
        dP = 0.02 * (dx_eff/Dh) * (rho * v_liq**2)/2 
        curr_Pc += dP # La pression augmente car on remonte le courant
        
        # Mise à jour UI
        if i % 30 == 0:
            progress_bar.progress((N_pts - i) / N_pts)
            
    progress_bar.progress(1.0)
    return res

# ==============================================================================
# 3. INTERFACE UTILISATEUR & GRAPHIQUES
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Simulation", "📊 Résultats", "📘 Méthodologie", "📥 Rapport Complet"])

if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None

# --- ONGLET 1 : Lancement & Visualisation Rapide ---
with tab1:
    st.write("### Contrôle de la Simulation")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Cliquez ci-dessous pour lancer le calcul.")
        launch_btn = st.button("▶️ LANCER LA SIMULATION", use_container_width=True, type="primary")

    if launch_btn:
        with st.spinner("Calculs thermodynamiques en cours (Équilibre + Transferts)..."):
            st.session_state['sim_data'] = run_simulation_logic()
    
    if st.session_state['sim_data']:
        data = st.session_state['sim_data']
        x_c, r_c, vel = data['x'], data['r'], data['Vel']
        st.success("Simulation terminée avec succès !")
        
        # --- GRAPHIQUE 1 : CHAMP DE VITESSE (2D) ---
        st.markdown("---")
        st.subheader("1. Champ de Vitesse Mach (2D)")
        st.caption("Visualisation du développement de l'écoulement supersonique dans le divergent.")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        Y_MAX = max(r_c) * 1.4
        y_vals = np.linspace(-Y_MAX, Y_MAX, 120)
        X_grid, Y_grid = np.meshgrid(x_c, y_vals)
        
        # Interpolation et masquage pour effet visuel "Tuyère"
        Vel_Mag_2D = np.zeros_like(X_grid)
        Mask_2D = np.ones_like(X_grid, dtype=bool)

        v_interp_1d = np.interp(X_grid[0,:], x_c, vel)
        r_interp_1d = np.interp(X_grid[0,:], x_c, r_c)
        v_interp_2d = np.tile(v_interp_1d, (len(y_vals), 1))
        r_interp_2d = np.tile(r_interp_1d, (len(y_vals), 1))
        dist_norm = np.abs(Y_grid) / r_interp_2d
        
        # Profil de vitesse turbulent approximé (loi en puissance 1/7)
        mask_valid = dist_norm <= 1.0
        Vel_Mag_2D[mask_valid] = v_interp_2d[mask_valid] * (1 - dist_norm[mask_valid]**2)**(1/7)
        Mask_2D[mask_valid] = False
        Vel_Mag_Ma = np.ma.masked_where(Mask_2D, Vel_Mag_2D)
        
        cmap = plt.get_cmap('turbo')
        mesh = ax1.pcolormesh(X_grid, Y_grid, Vel_Mag_Ma, cmap=cmap, shading='auto')
        plt.colorbar(mesh, ax=ax1, label="Vitesse (m/s)")
        
        # Ajout des vecteurs de flux (Quivers)
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

        # --- GRAPHIQUE 2 : PROFILS THERMIQUES ---
        st.markdown("---")
        st.subheader("2. Profils Thermiques aux Interfaces")
        st.caption("Coupes transversales de la température à travers les couches limites (Gaz -> Mur -> Liquide).")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = [20, data['idx_t'], len(x_c)-20] # Entrée, Col, Sortie
        labels = ["Entrée", "Col", "Sortie"]
        colors = ["green", "red", "blue"]
        
        for k, idx in enumerate(indices):
            Ti, Twh, Twc, Tliq = data['T_gas'][idx], data['T_wh'][idx], data['T_wc'][idx], data['T_cool'][idx]
            # Création de profils fictifs pour la visualisation des couches limites
            x_g, y_g = np.linspace(-0.5, 0, 15), Twh + (Ti - Twh)*(1 - np.exp(np.linspace(-0.5, 0, 15)/0.1))
            x_w, y_w = np.linspace(0, 1, 10), np.linspace(Twh, Twc, 10) # Conduction linéaire
            x_l, y_l = np.linspace(1, 1.5, 15), Tliq + (Twc - Tliq)*np.exp(-(np.linspace(1, 1.5, 15)-1)/0.1)
            
            full_x, full_y = np.concatenate([x_g, x_w, x_l]), np.concatenate([y_g, y_w, y_l])
            ax2.plot(full_x, full_y, color=colors[k], linewidth=2, label=labels[k])
            
            # Étiquettes de température
            bbox = dict(boxstyle="round,pad=0.3", fc="white", ec=colors[k], alpha=0.8)
            ax2.text(0, Twh + (60 if k==1 else -60), f"{int(Twh)} K", fontsize=9, color=colors[k], fontweight='bold', ha='right', bbox=bbox)
            ax2.text(1, Twc, f"{int(Twc)} K", fontsize=9, color=colors[k], fontweight='bold', ha='left', bbox=bbox)

        ax2.axvspan(0, 1, color='orange', alpha=0.1, label='Paroi')
        ax2.set_xlabel("Zone Normalisée (Gaz -> Paroi -> Liquide)")
        ax2.set_ylabel("Température (K)")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig2)

        # --- GRAPHIQUE 3 : VUE GLOBALE MULTI-AXES ---
        st.markdown("---")
        st.subheader("3. Vue d'Ensemble : Températures et Mach")
        
        fig3, host = plt.subplots(figsize=(12, 6))
        fig3.subplots_adjust(right=0.75) # Espace réservé pour les axes multiples

        par1 = host.twinx()
        par2 = host.twinx()
        par3 = host.twinx()

        # Décalage des axes supplémentaires vers la droite
        par2.spines["right"].set_position(("axes", 1.08))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        par3.spines["right"].set_position(("axes", 1.16))
        make_patch_spines_invisible(par3)
        par3.spines["right"].set_visible(True)

        # Tracés courbes
        p_geo, = host.plot(x_c, r_c*100, "k-", linewidth=3, alpha=0.4, label="Géométrie")
        host.plot(x_c, -r_c*100, "k-", linewidth=3, alpha=0.4)
        host.fill_between(x_c, r_c*100, -r_c*100, color='gray', alpha=0.1)

        p_gas, = par1.plot(x_c, data['T_gas'], "r-", label="T° Gaz")
        p_wall, = par1.plot(x_c, data['T_wh'], "orange", linestyle="--", label="T° Paroi (Chaude)")
        p_cool, = par2.plot(x_c, data['T_cool'], "b-", linewidth=2, label="T° Liquide")
        p_mach, = par3.plot(x_c, data['Mach'], "g-.", label="Mach")

        # Mise en forme des axes
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

        # --- GRAPHIQUE 4 : PRESSIONS ---
        st.markdown("---")
        st.subheader("4. Évolution des Pressions")
        st.caption("Comparaison de la détente des gaz (Pression motrice) et de la perte de charge du liquide.")
        
        fig4, ax_main = plt.subplots(figsize=(10, 5))
        ax_pg = ax_main.twinx() # Axe Pression Gaz
        ax_pc = ax_main.twinx() # Axe Pression Coolant
        
        # Décalage
        ax_pc.spines["right"].set_position(("axes", 1.12))
        make_patch_spines_invisible(ax_pc)
        ax_pc.spines["right"].set_visible(True)

        # Géométrie en fond pour contexte
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

        # --- GRAPHIQUE 5 : ZOOM RADIAL ---
        st.markdown("---")
        st.subheader("5. Zoom : Coupe Radiale dans le Divergent")
        
        # Sélection d'une section spécifique dans le divergent (x = 2 * diamètre col)
        D_t = 2 * min(r_c)
        target_x = 2 * D_t
        idx_target = (np.abs(x_c - target_x)).argmin()
        
        vals = {k: data[k][idx_target] for k in ['T_gas', 'T_wh', 'T_wc', 'T_cool']}
        
        # Génération profils micro pour le zoom
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
        
        # Annotations textuelles
        ax5.text(-0.4, vals['T_gas']-200, "GAZ", color='red', fontweight='bold')
        ax5.text(wall_thickness*1000/2, (vals['T_wh']+vals['T_wc'])/2, "PAROI", ha='center', fontweight='bold', rotation=90)
        ax5.text(wall_thickness*1000 + 0.1, vals['T_cool']+20, "H2 LIQUIDE", color='blue', fontweight='bold')

        ax5.set_xlabel("Distance (mm) - 0 = Interface Paroi/Gaz")
        ax5.set_ylabel("Température (K)")
        ax5.set_title
