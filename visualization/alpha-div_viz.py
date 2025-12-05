import numpy as np
import plotly.graph_objects as go

# --- 1. Définition de la Divergence D_alpha (Gestion Complète des Cas) ---

def D_alpha_complete(x, y, alpha):
    """
    Calcule la divergence D^(alpha)(x, y) en gérant explicitement les cas limites
    (alpha = 1, alpha = -1) et les domaines |alpha| > 1.
    """
    
    # 1. Préparation des données: Assurer x, y > 0 pour le logarithme et les puissances
    x_safe = np.where(x > 0, x, np.nan)
    y_safe = np.where(y > 0, y, np.nan)
    
    # Cas générique |alpha| < 1 (ou stablement |alpha| < 0.99)
    if np.abs(alpha) < 0.99:
        p = 2 / (1 - alpha)
        q = 2 / (1 + alpha)
        
        # Formule générale
        term1 = x_safe / p
        term2 = y_safe / q
        term3 = (x_safe**(1/p)) * (y_safe**(1/q))
        
        result = p * q * (term1 + term2 - term3)
        return result
        
    # 2. Cas Limite alpha = 1 (Approximation de Kullback-Leibler)
    elif np.isclose(alpha, 1.0):
        # D^(1)(x, y) = x log(x/y) - x + y
        with np.errstate(divide='ignore', invalid='ignore'):
            result = x_safe * np.log(x_safe / y_safe) - x_safe + y_safe
        return result
        
    # 3. Cas Limite alpha = -1 (Approximation de KL renversée)
    elif np.isclose(alpha, -1.0):
        # D^(-1)(x, y) = y log(y/x) - y + x
        with np.errstate(divide='ignore', invalid='ignore'):
            result = y_safe * np.log(y_safe / x_safe) - y_safe + x_safe
        return result
        
    # 4. Cas |alpha| > 1 (p ou q est négatif, le calcul de puissance doit fonctionner)
    else:
        # Note: Cette zone peut avoir des artefacts visuels ou des valeurs très grandes.
        p = 2 / (1 - alpha)
        q = 2 / (1 + alpha)

        term1 = x_safe / p
        term2 = y_safe / q
        
        # numpy gère les puissances fractionnaires négatives si la base est positive
        with np.errstate(invalid='ignore'):
            term3 = (x_safe**(1/p)) * (y_safe**(1/q))
        
        result = p * q * (term1 + term2 - term3)
        return result


# --- 2. Préparation des données pour Plotly ---

X_MIN, X_MAX = 0.1, 5
Y_MIN, Y_MAX = 0.1, 5
N_POINTS = 50
x_vals = np.linspace(X_MIN, X_MAX, N_POINTS)
y_vals = np.linspace(Y_MIN, Y_MAX, N_POINTS)
X, Y = np.meshgrid(x_vals, y_vals)

# Domaine d'alpha: de -5 à 5, comme demandé (les bords -1 et 1 seront inclus)
ALPHA_MIN, ALPHA_MAX = -4.9, 4.9 
N_FRAMES = 100 # Augmenté pour lisser la transition
alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, N_FRAMES)


# --- 3. Création des Frames pour l'Animation/Curseur ---

frames = []
# Déterminer les bornes Z pour une échelle de couleur fixe (basée sur le domaine [-1, 1])
Z_MAX_GLOBAL = 10 
Z_MIN_GLOBAL = 0

for alpha in alpha_values:
    Z = D_alpha_complete(X, Y, alpha)
    
    frame = go.Frame(
        data=[
            go.Surface(
                z=Z, 
                x=X, 
                y=Y, 
                colorscale='Viridis',
                cmin=Z_MIN_GLOBAL, 
                cmax=Z_MAX_GLOBAL, # Limité pour l'affichage, car |alpha|>1 explose Z.
                showscale=True
            )
        ],
        name=str(round(alpha, 3))
    )
    frames.append(frame)


# --- 4. Configuration de la Figure Initiale et du Curseur ---

Z_initial = D_alpha_complete(X, Y, 0) 

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
    title=f"Divergence D^(\u03B1)(x, y) avec \u03B1 \u2208 [{ALPHA_MIN:.1f}, {ALPHA_MAX:.1f}]",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='D(\u03B1)(x, y) (Tronqué à 10)',
        zaxis=dict(range=[Z_MIN_GLOBAL, Z_MAX_GLOBAL]), # Fixer l'axe Z
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