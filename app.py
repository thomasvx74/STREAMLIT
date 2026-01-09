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
# 1. CONFIGURATION GLOBALE ET UTILITAIRES D'AFFICHAGE
# =============================================================================
st.set_page_config(
    page_title="Simulateur Régénératif Avancé",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_patch_spines_invisible(ax):
    """
    Masque les bordures (spines) des axes supplémentaires pour les graphiques 
    à échelles multiples (twinx), améliorant ainsi la lisibilité.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# En-tête de l'application
st.title("🚀 Simulation Avancée : Tuyère & Refroidissement Régénératif")
st.markdown("""
Ce simulateur modélise les échanges thermiques couplés (Gaz-Paroi-Liquide) dans une tuyère de fusée.
Il intègre l'effet des canaux hélicoïdaux (en spirale) pour augmenter l'efficacité du refroidissement.
""")

# =============================================================================
# 2. INTERFACE UTILISATEUR : PARAMÈTRES D'ENTRÉE (SIDEBAR)
# =============================================================================
st.sidebar.header("1. Géométrie des Canaux")
# Paramètres géométriques des canaux de refroidissement (cooling channels)
N_channels = st.sidebar.number_input("Nombre de canaux", value=100, step=10)
channel_width_mm = st.sidebar.number_input("Largeur canal (mm)", value=1.5, step=0.1)
channel_height_mm = st.sidebar.number_input("Hauteur canal (mm)", value=4.0, step=0.1)
wall_thickness_mm = st.sidebar.number_input("Épaisseur paroi (mm)", value=1.0, step=0.1)
channel_angle = st.sidebar.slider("Angle d'hélice (°)", 0.0, 60.0, 0.0, step=5.0, 
                                  help="0° = canaux droits. Un angle augmente la surface d'échange.")

# Conversion des unités millimétriques en mètres (SI) pour les calculs physiques
channel_width = channel_width_mm / 1000.0
channel_height = channel_height_mm / 1000.0
wall_thickness = wall_thickness_mm / 1000.0

st.sidebar.header("2. Conditions Moteur")
# Paramètres thermodynamiques de la chambre de combustion et du fluide
P_cc_bar = st.sidebar.number_input("Pression Chambre (bar)", value=70.0, step=5.0)
P_cc = P_cc_bar * 1e5  # Conversion bar -> Pa
mdot_coolant = st.sidebar.number_input("Débit liquide (kg/s)", value=2.2, step=0.1)
T_coolant_in = st.sidebar.number_input("Temp. Entrée Liquide (K)", value=35.0, step=1.0)
P_coolant_in_bar = st.sidebar.number_input("Pression Liquide (bar)", value=110.0, step=5.0)
P_coolant_in = P_coolant_in_bar * 1e5 # Conversion bar -> Pa

st.sidebar.header("3. Matériau")
k_material = st.sidebar.number_input("Conductivité (W/m.K)", value=390.0, 
                                     help="390 pour le Cuivre (très conducteur), 20 pour l'Inconel (résistant mais isolant)")

# =============================================================================
# 3. MOTEUR PHYSIQUE : FONCTIONS DE CALCUL
# =============================================================================

@st.cache_data
def generate_geometry():
    """
    Génère ou charge le profil géométrique de la tuyère (Rayon vs Position X).
    Si le fichier 'profil_tuyere.csv' n'existe pas, un profil standard de Laval est créé.
    
    Returns:
        tuple: Tableaux numpy des positions axiales (x) et des rayons (r).
    """
    csv_file = "profil_tuyere.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df['x'].values, df['radius'].values
    else:
        # Création d'un profil parabolique standard
        x = np.linspace(-0.05, 0.25, 300)
        # Formule conditionnelle : Convergent vs Divergent
        r = np.where(x < 0, 0.03 + x**2 * 8, 0.03 + 0.12 * np.sqrt(np.abs(x)))
        pd.DataFrame({'x': x, 'radius': r}).to_csv(csv_file, index=False)
        return x, r

def get_mach(area_ratio, gamma, subsonic=True):
    """
    Résout la relation Aire-Mach pour un écoulement isentropique.
    
    Args:
        area_ratio (float): Ratio A/A_col.
        gamma (float): Coefficient adiabatique du gaz.
        subsonic (bool): True pour la solution subsonique (convergent), False pour supersonique (divergent).
    
    Returns:
        float: Nombre de Mach local.
    """
    def eq(M):
        if M <= 0: return 1e6 # Évite les erreurs mathématiques
        # Équation de Saint-Venant pour les écoulements compressibles
        return (1/M)*((2/(gamma+1))*(1+(gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) - area_ratio
    
    guess = 0.1 if subsonic else 2.5
    return fsolve(eq, guess)[0]

def bartz_hg(D_t, P_cc, c_star, At_A, T_cc, T_w, gamma, M, mu_g, cp_g, Pr_g):
    """
    Calcule le coefficient de convection gaz (hg) via la corrélation de Bartz simplifiée.
    Cette corrélation est standard pour les moteurs-fusées à ergols liquides.
    """
    # Facteur de correction sigma (prend en compte la température de paroi)
    sigma = ((0.5 * T_w/T_cc * (1 + (gamma-1)/2 * M**2) + 0.5)**(-0.68) * (1 + (gamma-1)/2 * M**2)**(-0.12))
    # Formule de Bartz
    hg = (0.026 / (D_t**0.2)) * (mu_g**0.2 * cp_g / Pr_g**0.6) * (P_cc/c_star)**0.8 * (At_A)**0.9 * sigma
    return hg

def run_simulation_logic():
    """
    Fonction principale exécutant la boucle de simulation quasi-1D.
    Couple l'aérodynamique, la combustion (Cantera), les fluides réels (CoolProp) et le transfert thermique.
    """
    
    # --- A. Initialisation Géométrique ---
    x_coords, r_coords = generate_geometry()
    dr_dx = np.gradient(r_coords, x_coords) # Pente de la paroi
    A = np.pi * r_coords**2                 # Aires de section
    idx_t = np.argmin(A)                    # Index du col (throat)
    A_t, D_t = A[idx_t], 2*r_coords[idx_t]

    # --- B. Calcul Thermodynamique du Gaz (Cantera) ---
    try:
        # Utilisation du mécanisme GRI-3.0 pour simuler la combustion H2/O2
        gas = ct.Solution('gri30.yaml')
        gas.TPX = 300, P_cc, {'H2': 1, 'O2': 0.5} 
        gas.equilibrate('HP') # Équilibre à Enthalpie et Pression constantes
        
        # Récupération des propriétés du mélange brûlé
        T_cc, gamma = gas.T, gas.cp/gas.cv
        mu_g, cp_g, Pr_g = gas.viscosity, gas.cp, 0.7
        R_spec = ct.gas_constant / gas.mean_molecular_weight
        c_star = P_cc * A_t * np.sqrt(gamma) # Vitesse caractéristique
    except Exception as e:
        st.error(f"Erreur Cantera (vérifiez la présence de gri30.yaml) : {e}")
        return None

    # --- C. Préparation des Tableaux de Résultats ---
    N_pts = len(x_coords)
    res = {
        'x': x_coords, 'r': r_coords, 'dr_dx': dr_dx, 'idx_t': idx_t,
        'Vel': np.zeros(N_pts), 'Mach': np.zeros(N_pts),
        'P_gas': np.zeros(N_pts), 'P_cool': np.zeros(N_pts),
        'T_gas': np.zeros(N_pts),
        'T_wh': np.zeros(N_pts), 'T_wc': np.zeros(N_pts),
        'T_cool': np.zeros(N_pts)
    }

    # --- D. Boucle de Résolution Spatiale (Contre-Courant) ---
    # Le liquide entre par la sortie de la tuyère (nozzle exit) et remonte vers la chambre.
    # On itère donc de la fin (i=N_pts) vers le début (i=0).
    
    curr_Tc = T_coolant_in
    curr_Pc = P_coolant_in
    
    # Prise en compte de la géométrie hélicoïdale des canaux
    angle_rad = np.radians(channel_angle)
    angle_factor = 1.0 / np.cos(angle_rad) # Augmentation de la longueur effective

    progress_bar = st.progress(0)

    for i in range(N_pts-1, -1, -1):
        # 1. Aérodynamique Gaz (Isentropique)
        is_sub = (i < idx_t) # Subsonique avant le col, Supersonique après
        M = get_mach(A[i]/A_t, gamma, subsonic=is_sub)
        res['Mach'][i] = M
        
        # 2. Propriétés Statiques du Gaz
        T_st = T_cc / (1 + (gamma-1)/2 * M**2)
        P_st = P_cc / (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))
        res['P_gas'][i] = P_st
        
        # Température adiabatique de paroi (Recovery Temperature)
        T_aw = T_st * (1 + 0.9 * (gamma-1)/2 * M**2)
        res['Vel'][i] = M * np.sqrt(gamma * R_spec * T_st)

        # 3. Propriétés du Liquide de Refroidissement (CoolProp)
        # Gestion des fluides réels (ex: Hydrogène supercritique)
        try:
            rho = PropsSI('D','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            mu  = PropsSI('V','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            cp  = PropsSI('C','P',curr_Pc,'T',curr_Tc,'Hydrogen')
            k_l = PropsSI('L','P',curr_Pc,'T',curr_Tc,'Hydrogen')
        except:
            # Valeurs par défaut en cas de défaillance CoolProp (ex: hors bornes)
            rho, mu, cp, k_l = 70, 1e-5, 14000, 0.1

        # 4. Coefficients d'Échange Thermique
        W, H = channel_width, channel_height
        Dh = 4*W*H / (2*(W+H)) # Diamètre hydraulique
        
        # Côté Liquide : Corrélation Dittus-Boelter ou similaire
        Re = (mdot_coolant/(N_channels*W*H)) * Dh / mu
        h_c = 0.023 * (k_l/Dh) * Re**0.8 * (cp*mu/k_l)**0.4
        
        # Efficacité des ailettes (Fin Efficiency)
        # La paroi entre les canaux agit comme une ailette
        t_fin = (2*np.pi*r_coords[i] - N_channels*W) / N_channels
        if t_fin < 1e-5: t_fin = 1e-5
        m_fin = np.sqrt(2 * h_c / (k_material * t_fin))
        eta_f = np.tanh(m_fin * H) / (m_fin * H)
        h_c_eff = h_c * (W + 2 * H * eta_f) / (W + t_fin) # Coefficient effectif augmenté
        
        # Côté Gaz : Corrélation de Bartz
        h_g = bartz_hg(D_t, P_cc, c_star, A_t/A[i], T_cc, 800, gamma, M, mu_g, cp_g, Pr_g)
        
        # 5. Bilan Thermique (Analogie Électrique)
        # R_tot = R_conv_gaz + R_cond_paroi + R_conv_liquide
        R_tot = (1/h_g) + (wall_thickness/k_material) + (1/(h_c_eff * angle_factor))
        q = (T_aw - curr_Tc) / R_tot # Flux thermique (W/m2)
        
        # Calcul des températures intermédiaires
        res['T_gas'][i] = T_aw
        res['T_wh'][i]  = T_aw - q/h_g                  # T° Paroi côté gaz
        res['T_wc'][i]  = res['T_wh'][i] - q*(wall_thickness/k_material) # T° Paroi côté liquide
        res['T_cool'][i]= curr_Tc
        res['P_cool'][i]= curr_Pc
        
        # 6. Intégration pour le pas suivant (dx)
        dx = abs(x_coords[i] - x_coords[i-1]) if i > 0 else 0.001
        dx_eff = dx * angle_factor
        
        # Mise à jour enthalpique du liquide (Échauffement)
        curr_Tc += q * (2*np.pi*r_coords[i]*dx) / (mdot_coolant * cp)
        
        # Mise à jour hydraulique (Perte de charge)
        # Approximation Darcy-Weisbach
        v_liq = mdot_coolant / (rho * N_channels * W * H)
        dP = 0.02 * (dx_eff/Dh) * (rho * v_liq**2)/2 
        curr_Pc += dP # La pression augmente car on remonte le courant
        
        if i % 30 == 0:
            progress_bar.progress((N_pts - i) / N_pts)
            
    progress_bar.progress(1.0)
    return res

# =============================================================================
# 4. VISUALISATION ET DASHBOARD
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Simulation", "📊 Résultats", "📘 Méthodologie", "📥 Rapport Complet"])

if 'sim_data' not in st.session_state:
    st.session_state['sim_data'] = None

# --- ONGLET 1 : Lancement et Graphiques Principaux ---
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
        # GRAPHIQUE 1 : CHAMP DE VITESSE (Approximation 2D)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("1. Champ de Vitesse Mach (2D)")
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        Y_MAX = max(r_c) * 1.4
        y_vals = np.linspace(-Y_MAX, Y_MAX, 120)
        X_grid, Y_grid = np.meshgrid(x_c, y_vals)
        
        # Création du masque pour ne dessiner que dans la tuyère
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
        
        # Rendu visuel
        cmap = plt.get_cmap('turbo')
        mesh = ax1.pcolormesh(X_grid, Y_grid, Vel_Mag_Ma, cmap=cmap, shading='auto')
        plt.colorbar(mesh, ax=ax1, label="Vitesse (m/s)")
        
        # Ajout des vecteurs vitesse (Quiver) pour visualiser l'écoulement
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
        # GRAPHIQUE 2 : PROFILS THERMIQUES INTERFACIAUX
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Profils Thermiques aux Interfaces")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = [20, data['idx_t'], len(x_c)-20] # Entrée, Col, Sortie
        labels = ["Entrée", "Col", "Sortie"]
        colors = ["green", "red", "blue"]
        
        for k, idx in enumerate(indices):
            Ti, Twh, Twc, Tliq = data['T_gas'][idx], data['T_wh'][idx], data['T_wc'][idx], data['T_cool'][idx]
            # Interpolation visuelle pour représenter les couches limites
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
        # GRAPHIQUE 3 : VUE GLOBALE MULTI-AXES
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("3. Vue d'Ensemble : Températures et Mach")
        
        fig3, host = plt.subplots(figsize=(12, 6))
        fig3.subplots_adjust(right=0.75) # Marge pour les axes multiples

        par1 = host.twinx()
        par2 = host.twinx()
        par3 = host.twinx()

        # Positionnement des axes décalés
        par2.spines["right"].set_position(("axes", 1.08))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        par3.spines["right"].set_position(("axes", 1.16))
        make_patch_spines_invisible(par3)
        par3.spines["right"].set_visible(True)

        # Tracés des courbes
        p_geo, = host.plot(x_c, r_c*100, "k-", linewidth=3, alpha=0.4, label="Géométrie")
        host.plot(x_c, -r_c*100, "k-", linewidth=3, alpha=0.4)
        host.fill_between(x_c, r_c*100, -r_c*100, color='gray', alpha=0.1)

        p_gas, = par1.plot(x_c, data['T_gas'], "r-", label="T° Gaz")
        p_wall, = par1.plot(x_c, data['T_wh'], "orange", linestyle="--", label="T° Paroi (Chaude)")
        p_cool, = par2.plot(x_c, data['T_cool'], "b-", linewidth=2, label="T° Liquide")
        p_mach, = par3.plot(x_c, data['Mach'], "g-.", label="Mach")

        # Configuration des labels et couleurs
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
        # GRAPHIQUE 4 : PERTES DE CHARGE ET PRESSIONS
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("4. Évolution des Pressions")
        
        fig4, ax_main = plt.subplots(figsize=(10, 5))
        ax_pg = ax_main.twinx()
        ax_pc = ax_main.twinx()
        
        # Décalage axe
        ax_pc.spines["right"].set_position(("axes", 1.12))
        make_patch_spines_invisible(ax_pc)
        ax_pc.spines["right"].set_visible(True)

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
        # GRAPHIQUE 5 : ZOOM SUR LA COUCHE LIMITE (Divergent)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("5. Zoom : Coupe Radiale dans le Divergent")
        
        # Sélection d'une section spécifique (x = 2 * diamètre col)
        D_t = 2 * min(r_c)
        target_x = 2 * D_t
        idx_target = (np.abs(x_c - target_x)).argmin()
        
        vals = {k: data[k][idx_target] for k in ['T_gas', 'T_wh', 'T_wc', 'T_cool']}
        
        # Génération des profils de température pour le zoom
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
        
        ax5.text(-0.4, vals['T_gas']-200, "GAZ", color='red', fontweight='bold')
        ax5.text(wall_thickness*1000/2, (vals['T_wh']+vals['T_wc'])/2, "PAROI", ha='center', fontweight='bold', rotation=90)
        ax5.text(wall_thickness*1000 + 0.1, vals['T_cool']+20, "H2 LIQUIDE", color='blue', fontweight='bold')

        ax5.set_xlabel("Distance (mm) - 0 = Interface Paroi/Gaz")
        ax5.set_ylabel("Température (K)")
        ax5.set_title(f"Profil Radial à x={x_c[idx_target]:.3f} m")
        ax5.grid(True, linestyle=':')
        ax5.legend()
        
        st.pyplot(fig5)

# --- ONGLET 2 : EXPORTATION DES DONNÉES ---
with tab2:
    st.header("📋 Tableau de Données Brutes")
    
    if st.session_state['sim_data']:
        data = st.session_state['sim_data']
        
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
        
        st.dataframe(df_results, use_container_width=True)
        
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
with tab3:
    st.markdown("## 📘 Architecture & Méthodologie du Code")
    
    st.info("""
    **Résumé du Fonctionnement :** Ce simulateur repose sur une approche **Quasi-1D Stationnaire**.  
    La tuyère est découpée en une centaine de "tranches" (discrétisation spatiale). Pour chaque tranche, le code résout les équations de conservation de la masse, de l'énergie et de la quantité de mouvement pour déterminer l'équilibre thermique entre le gaz brûlant et le liquide de refroidissement.
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 1. ARCHITECTURE LOGICIELLE (Amélioré)
    # -------------------------------------------------------------------------
    st.header("1. Le Moteur de Calcul")
    st.write("La précision du modèle repose sur l'utilisation de bibliothèques thermodynamiques de référence, évitant les approximations constantes.")

    col_lib1, col_lib2, col_lib3 = st.columns([1, 1, 1])

    with col_lib1:
        st.error("**🧪 Cantera**")
        st.caption("Chimie & Combustion")
        st.markdown("""
        *Gère le côté "Feu"*
        * Calcule l'équilibre chimique complexe (dissociation).
        * Fournit le $C_p$, $\gamma$ et la T° de flamme exacts.
        """)

    with col_lib2:
        st.info("**❄️ CoolProp**")
        st.caption("Fluides Réels")
        st.markdown("""
        *Gère le côté "Glace"*
        * Indispensable pour l'Hydrogène supercritique/liquide.
        * Calcule les pics de chaleur spécifique ($C_p$) près du point critique.
        """)
    
    with col_lib3:
        st.warning("**🐍 SciPy & NumPy**")
        st.caption("Mathématiques")
        st.markdown("""
        *Le Cerveau*
        * Résolution des équations non-linéaires (Newton-Raphson) pour trouver le Mach.
        * Gestion vectorielle des données.
        """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. ALGORITHME VISUEL (Graphviz Avancé)
    # -------------------------------------------------------------------------
    st.header("2. Logique de Résolution")
    st.write("Le système est un **échangeur de chaleur à contre-courant**. Le gaz va de gauche à droite, le liquide de droite à gauche.")

    # Diagramme Graphviz amélioré avec clusters et flow visuel
    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        compound=true;
        node [fontname="Helvetica", fontsize=10, style="filled,rounded", shape=box];
        edge [color="#666666", arrowsize=0.8];

        # Entrées
        subgraph cluster_input {
            label = "INITIALISATION";
            style=dashed; color="#bdbdbd"; fontcolor="#bdbdbd";
            Geom [label="📐 Géométrie\n(Profil .csv)", fillcolor="#e0e0e0"];
            Comb [label="🔥 Combustion\n(P_chambre, Ratio O/F)", fillcolor="#ffccbc"];
        }

        # Cœur du calcul
        subgraph cluster_main {
            label = "BOUCLE DE CALCUL (Par tranche dx)";
            style=filled; bgcolor="#f9fbe7"; color="#c5e1a5"; fontcolor="#558b2f";
            
            # Côté Gaz
            subgraph cluster_gas {
                label = "Côté Gaz (Chaud)";
                style=filled; bgcolor="#ffab91"; color="#ff7043"; fontcolor="white";
                Mach [label="1. Aérodynamique\nMach = f(Area)", fillcolor="#ffccbc"];
                Bartz [label="2. Convection Gaz\nCorrélation de Bartz\n(Calcul h_g)", fillcolor="#ffab91"];
            }

            # Mur
            Wall [label="3. Conduction Paroi\n(Cuivre / Inconel)", shape=rect, fillcolor="#8d6e63", fontcolor="white", width=2];

            # Côté Liquide
            subgraph cluster_liq {
                label = "Côté Liquide (Froid)";
                style=filled; bgcolor="#81d4fa"; color="#0288d1"; fontcolor="white";
                Prop [label="4. Propriétés Fluide\n(CoolProp @ P, T)", fillcolor="#b3e5fc"];
                Dittus [label="5. Convection Liq.\nCorrélation Dittus-Boelter\n(Calcul h_l)", fillcolor="#81d4fa"];
            }

            Balance [label="⚖️ BILAN FLUX\nConvergence T_paroi", shape=diamond, fillcolor="#fff176", style="filled,dashed"];
        }

        # Sortie
        Output [label="📊 VISUALISATION\nGraphiques & KPIs", shape=folder, fillcolor="#a5d6a7"];

        # Liaisons
        Geom -> Mach;
        Comb -> Mach;
        
        Mach -> Bartz;
        Bartz -> Wall [label="Flux entrant"];
        
        Prop -> Dittus;
        Dittus -> Wall [label="Flux sortant"];
        
        Wall -> Balance;
        Balance -> Prop [label="Mise à jour T_liq\n(Pas suivant)", style=dashed, dir=back];
        
        Balance -> Output [label="Fin de boucle"];
    }
    ''')
    
    st.caption("Note : Le calcul du fluide de refroidissement s'effectue souvent à rebours (de la sortie vers l'injecteur) pour correspondre à la physique du contre-courant.")

    st.markdown("---") 

    # -------------------------------------------------------------------------
    # 3. PHYSIQUE DÉTAILLÉE
    # -------------------------------------------------------------------------
    st.header("3. Modèles Physiques Utilisés")

    st.markdown("""
    Pour estimer les températures, nous modélisons le transfert de chaleur comme un **réseau de résistances électriques**. La chaleur ("le courant") doit traverser trois obstacles successifs.
    """)

    

    # --- A. Le Circuit Thermique ---
    st.subheader("A. L'Équation Maîtresse du Flux")
    st.latex(r"""
    Q_{flux} = \frac{T_{gaz}^{adiabatique} - T_{liquide}}{ R_{convection\_gaz} + R_{conduction\_paroi} + R_{convection\_liquide} }
    """)
    
    # --- B. Détail des Corrélations (Expanders pour ne pas surcharger) ---
    c1, c2 = st.columns(2)
    
    with c1:
        with st.expander("🔥 Côté Gaz : Bartz"):
            st.write("La convection des gaz à haute vitesse est estimée par la formule de **Bartz** (simplifiée ici) :")
            st.latex(r"""
            h_g = \frac{0.026}{D^{0.2}} \left( \frac{\mu^{0.2} C_p}{Pr^{0.6}} \right) \left( \frac{P_c}{c^*} \right)^{0.8} \sigma
            """)
            st.write("""
            **Ce qu'il faut retenir :**
            * Le transfert est maximal au **Col** (là où le diamètre $D$ est petit).
            * Il augmente avec la Pression Foyer ($P_c$).
            """)

    with c2:
        with st.expander("❄️ Côté Liquide : Dittus-Boelter"):
            st.write("Pour le refroidissement dans les canaux, on utilise des corrélations classiques de Nusselt ($Nu$) type **Dittus-Boelter** ou **Gnielinski** :")
            st.latex(r"""
            Nu = 0.023 Re^{0.8} Pr^{0.4}
            """)
            st.write("""
            **Ce qu'il faut retenir :**
            * La vitesse du fluide ($Re$) est la clé : plus ça circule vite, mieux ça refroidit.
            * Si le liquide bout, le transfert change radicalement (non géré dans ce modèle simple).
            """)

    st.markdown("---")
    
    # --- C. Géométrie des Canaux ---
    st.subheader("B. Géométrie des Canaux & Ailettes")
    st.write("Le code prend en compte l'augmentation de la surface d'échange due aux canaux (effet d'ailette).")
    
    

    st.info("""
    **Efficacité des Ailettes ($\eta$) :** Les parois latérales des canaux ("ribs") aident à évacuer la chaleur. Le code calcule un rendement d'ailette pour ne pas surestimer le refroidissement, car le haut de l'ailette est plus chaud que la base.
    """)
# --- ONGLET 4 : GESTION DES RAPPORTS PDF ---
with tab4:
    st.header("📄 Rapport Technique")
    nom_du_fichier = "Rapport_TT.pdf"
    
    import base64
    def show_pdf(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    try:
        with open(nom_du_fichier, "rb") as f:
            pdf_data = f.read()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Télécharger le Rapport (PDF)",
                data=pdf_data,
                file_name="Rapport_Tuyere_Rocket.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col2:
            show_pdf(nom_du_fichier)
    except FileNotFoundError:
        st.error(f"⚠️ Erreur : Le fichier '{nom_du_fichier}' est introuvable.")

