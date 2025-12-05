import numpy as np
import plotly.graph_objects as go

# --- 1. Fonction Psi_alpha en 1D (vectorisée pour la grille) ---

def Psi_alpha_Surface(M, A):
    """
    Calcule Psi_alpha(mu) pour chaque point de la grille (mu, alpha).
    M et A sont des matrices (meshgrid).
    """
    # 1. Préparation et Masques
    M = np.where(M > 0, M, np.nan) # Mu doit être > 0
    Z = np.full_like(M, np.nan)
    
    # Masques pour les régions critiques et générales
    mask_neg_1 = (np.abs(A + 1.0) < 0.02) # Région autour de alpha = -1
    mask_pos_1 = (np.abs(A - 1.0) < 0.02) # Région autour de alpha = 1
    mask_inf = (np.abs(A) >= 3.9)         # Région |alpha| = inf (proxy)
    
    mask_gen_lt1 = (np.abs(A) < 1.0)      # |alpha| < 1
    mask_gen_gt1 = (np.abs(A) > 1.0) & (np.abs(A) < 3.9) # 1 < |alpha| < inf

    # 2. Calcul Z avec gestion des erreurs
    with np.errstate(divide='ignore', invalid='ignore'):
        
        # --- Cas Généraux : Puissance ---
        P = 2.0 / (1.0 - A)
        
        # A. Région |alpha| < 1 : Psi = (1/p) * mu^p
        mask = mask_gen_lt1
        p_val = P[mask]
        Z[mask] = (1.0 / p_val) * (M[mask]**p_val)

        # B. Région |alpha| > 1 : Psi = -(1/p) * mu^p
        mask = mask_gen_gt1
        p_val = P[mask]
        Z[mask] = - (1.0 / p_val) * (M[mask]**p_val)
        
        # --- Cas Limites (Overwriting) ---
        
        # C.1. Alpha = 1 (Exponentielle)
        mask = mask_pos_1
        Z[mask] = np.exp(M[mask])
        
        # C.2. Alpha = -1 (Entropie Négative)
        mask = mask_neg_1
        Z[mask] = M[mask] * np.log(M[mask]) - M[mask]
        
        # C.3. Alpha = INF (Logarithmique Négative)
        mask = mask_inf
        Z[mask] = -np.log(M[mask])

        # Final cleanup for NaN/Inf values
        Z[np.isinf(Z)] = np.nan
        Z = np.where(Z > 5, 5, Z) # Clipping Z
        
    return Z

# --- 2. Configuration du Domaine ---
MU_MIN, MU_MAX = 0.01, 3.0
ALPHA_MIN, ALPHA_MAX = -4.0, 4.0
N = 50
Z_MAX_LIMIT = 5 # Limite de hauteur de l'axe Z (très stricte pour voir le comportement près de 1)

# Création du Meshgrid
A_array = np.linspace(ALPHA_MIN, ALPHA_MAX, N) # Y-axis
M_array = np.linspace(MU_MIN, MU_MAX, N)      # X-axis
M, A = np.meshgrid(M_array, A_array) # Note: M est mu (X-axis), A est alpha (Y-axis)

# Calcul de la surface Z
Z = Psi_alpha_Surface(M, A)
Z = np.where(Z > Z_MAX_LIMIT, Z_MAX_LIMIT, Z) # Clipping final des valeurs trop grandes

# --- 3. Création du Graphique Plotly ---
fig = go.Figure(data=[go.Surface(z=Z, x=M, y=A, colorscale='viridis')])

fig.update_layout(
    title=r'Surface 3D de la Fonction Génératrice $\Psi_{\alpha}(\mu)$',
    scene=dict(
        xaxis_title=r'$\mu$',
        yaxis_title=r'$\alpha$',
        zaxis_title=r'$\Psi(\mu, \alpha)$ (Max 5)',
        zaxis=dict(range=[np.min(Z[~np.isnan(Z)]), Z_MAX_LIMIT]), # Ajustement de Z_min
        
        # Maintien de l'échelle visuelle
        aspectmode='manual',
        aspectratio=dict(x=1, y=4/3, z=1), # Ajustement visuel basé sur les plages de A (8) et M (3)
        
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=0.8)
        )
    )
)

fig.show()