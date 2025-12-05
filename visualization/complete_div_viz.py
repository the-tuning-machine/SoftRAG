import numpy as np
import plotly.graph_objects as go

# --- Fonctions Auxiliaires pour p et q ---

def calculate_p_q(alpha):
    """Calcule p et q, retourne np.nan si alpha est trop proche de +/- 1."""
    # S'assurer qu'il n'y a pas de division par zéro
    if np.isclose(alpha, 1.0) or np.isclose(alpha, -1.0):
        return np.nan, np.nan
    p = 2 / (1 - alpha)
    q = 2 / (1 + alpha)
    return p, q

# --- Fonctions de Transformation f_alpha et g_alpha (Optimisé pour la robustesse) ---

def f_alpha_g_alpha(input_x, input_y, alpha):
    """Applique les fonctions de transformation f_alpha(x) et g_alpha(y)."""
    
    X_out = np.full_like(input_x, np.nan, dtype=float)
    Y_out = np.full_like(input_y, np.nan, dtype=float)
    
    # 1. Troncature des entrées à 0.001 pour éviter les log/puissances de zéro
    x_safe = np.where(input_x > 0.001, input_x, 0.001)
    y_safe = np.where(input_y > 0.001, input_y, 0.001)
    
    # Cas générique |alpha| < 1
    if np.abs(alpha) < 1:
        p, q = calculate_p_q(alpha)
        with np.errstate(invalid='ignore'):
            X_out = x_safe**p
            Y_out = y_safe**q
            
    # Cas alpha = 1
    elif np.isclose(alpha, 1.0):
        X_out = np.exp(x_safe)
        Y_out = y_safe
            
    # Cas alpha = -1
    elif np.isclose(alpha, -1.0):
        X_out = x_safe
        Y_out = np.exp(y_safe)
        
    # Cas |alpha| > 1 (Utilisation de la puissance sur la base positive pour éviter les nombres complexes)
    else:
        p, q = calculate_p_q(alpha)
        # Gestion des cas très éloignés (pour simplifier f_alpha(x)=1/x et g_alpha(y)=-1/y)
        if alpha > 4.5: # Approximation de alpha = inf
             X_out = x_safe # f_alpha(x) = x
             Y_out = -1 / y_safe # g_alpha(y) = -1/y
        elif alpha < -4.5: # Approximation de alpha = -inf
             X_out = 1 / x_safe # f_alpha(x) = 1/x
             Y_out = -y_safe # g_alpha(y) = -y
        else:
             # Utilisation de la puissance simple pour les cas intermédiaires |alpha| > 1
             with np.errstate(invalid='ignore'):
                X_out = x_safe**p
                Y_out = y_safe**q
                
    return X_out, Y_out

# --- 1. Définition de la Divergence D_alpha (Mise à jour pour np.isclose) ---

def D_alpha_complete(x, y, alpha):
    """Calcule la divergence D^(alpha)(x, y)."""
    
    # 1. Assurer que les entrées (sorties de f et g) sont strictement positives
    x_safe = np.where(x > 0.001, x, 0.001)
    y_safe = np.where(y > 0.001, y, 0.001)
    
    # 2. Cas Limite alpha = 1 (Kullback-Leibler)
    if np.isclose(alpha, 1.0):
        # D^(1)(x, y) = x log(x/y) - x + y
        with np.errstate(divide='ignore', invalid='ignore'):
            result = x_safe * np.log(x_safe / y_safe) - x_safe + y_safe
        return result
        
    # 3. Cas Limite alpha = -1 (KL renversée)
    elif np.isclose(alpha, -1.0):
        # D^(-1)(x, y) = y log(y/x) - y + x
        with np.errstate(divide='ignore', invalid='ignore'):
            result = y_safe * np.log(y_safe / x_safe) - y_safe + x_safe
        return result
        
    # 4. Cas général (y compris |alpha| > 1)
    else:
        p, q = calculate_p_q(alpha)
        
        # Le calcul échoue si p ou q est NaN (ce qui signifie alpha est très proche de +/- 1)
        if np.isnan(p) or np.isnan(q):
            return np.full_like(x, np.nan)
        
        term1 = x_safe / p
        term2 = y_safe / q
        
        # Utilisation de np.errstate pour ignorer les avertissements des puissances
        with np.errstate(invalid='ignore'):
            term3 = (x_safe**(1/p)) * (y_safe**(1/q))
        
        result = p * q * (term1 + term2 - term3)
        return result


# --- 2. Préparation des données pour Plotly ---

# Domaine d'entrée stable
X_MIN, X_MAX = 0.1, 5
Y_MIN, Y_MAX = 0.1, 5
N_POINTS = 50
x_vals = np.linspace(X_MIN, X_MAX, N_POINTS)
y_vals = np.linspace(Y_MIN, Y_MAX, N_POINTS)
X, Y = np.meshgrid(x_vals, y_vals)

ALPHA_MIN, ALPHA_MAX = -4.9, 4.9 
N_FRAMES = 100
alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, N_FRAMES)

# Bornes Z fixes (ajustées pour la stabilité)
Z_MAX_GLOBAL = 10 
Z_MIN_GLOBAL = 0


# --- 3. Création des Frames pour l'Animation/Curseur ---

frames = []

for alpha in alpha_values:
    # 1. Transformation des entrées X et Y
    X_trans, Y_trans = f_alpha_g_alpha(X, Y, alpha)
    
    # 2. Calcul de la divergence D_alpha(f_alpha(X), g_alpha(Y))
    Z = D_alpha_complete(X_trans, Y_trans, alpha)
    
    frame = go.Frame(
        data=[
            go.Surface(
                z=Z, 
                x=X, 
                y=Y, 
                colorscale='Viridis',
                cmin=Z_MIN_GLOBAL, 
                cmax=Z_MAX_GLOBAL,
                showscale=True
            )
        ],
        name=str(round(alpha, 3))
    )
    frames.append(frame)


# --- 4. Configuration de la Figure Initiale et du Curseur ---

X_init_trans, Y_init_trans = f_alpha_g_alpha(X, Y, 0)
Z_initial = D_alpha_complete(X_init_trans, Y_init_trans, 0)

fig = go.Figure(
    data=[
        go.Surface(
            z=Z_initial,
            x=X,
            y=Y,
            colorscale='Viridis',
            cmin=Z_MIN_GLOBAL, 
            cmax=Z_MAX_GLOBAL,
            showscale=True
        )
    ],
    frames=frames
)

fig.update_layout(
    title=f"Divergence D^(\u03B1) (f_\u03B1(x), g_\u03B1(y)) avec \u03B1 \u2208 [{ALPHA_MIN:.1f}, {ALPHA_MAX:.1f}] (Optimisé)",
    scene=dict(
        xaxis_title='x (Entrée originale)',
        yaxis_title='y (Entrée originale)',
        zaxis_title='D(\u03B1)(f(\u03B1)(x), g(\u03B1)(y)) (Tronqué à 10)',
        zaxis=dict(range=[Z_MIN_GLOBAL, Z_MAX_GLOBAL]),
        aspectratio=dict(x=1, y=1, z=0.5),
    ),
    sliders=[{
        'active': N_FRAMES // 2,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': '\u03B1 = ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100, 'easing': 'linear'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'steps': [
            {
                'label': str(round(alpha_values[k], 2)), 
                'method': 'animate',
                'args': [
                    [frame.name], 
                    {'mode': 'immediate', 'frame': {'duration': 100, 'redraw': True}, 'transition': {'duration': 100}}
                ]
            } for k, frame in enumerate(frames)
        ]
    }],
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate', 'fromcurrent': True}]
        }]
    }]
)

# Affichage de la figure
fig.show()