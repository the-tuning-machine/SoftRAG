import numpy as np
from numba import njit
# from scipy.optimize import brentq # Nécessite SciPy pour les cas généraux

# --- Fonction Utilitaires pour Calculer les Exposants ---

@njit
def get_q(alpha):
  """Calcule l'exposant conjugué q tel que 1/p + 1/q = 1, où p = 1/(1-alpha)."""
  if np.isclose(alpha, 0): # L2
    return 2.0
  p = 1.0 / (1.0 - alpha)
  if np.isclose(p, 1.0): # Cas alpha=0
    return 2.0 
  if np.isclose(p, np.inf) or np.isclose(p, 0.0): # Cas alpha=1 ou alpha non défini
    return 0.0 # Non utilisé pour ces cas
  return p / (p - 1.0)


# --- Fonction pour le Gradient de Psi*_alpha ---

@njit
def psi_star_prime(u, alpha):
  """Calcule le gradient (dérivée) de Psi*_alpha(u) où u = s_i - gamma."""
  
  # Cas spéciaux avec forme fermée ou dérivée simple
  
  if np.isclose(alpha, 0): # Cas Quadratique (L2): |alpha|<1, q=2. Psi*'(u) = u
    return u
  
  # Si alpha -> 1 (Entropique): Psi*'(u) = exp(u). u = s_i - gamma
  # La condition de racine est incorrecte pour ce cas; on utilise la LSE-formule.
  # Nous retournons la dérivée pour l'équation si besoin d'un solveur général.
  if np.isclose(alpha, 1): 
    return np.exp(u)
  
  if np.isclose(alpha, -1): # Cas KL: |alpha|<1 (mais souvent traité à part), q=0.5 (non). Psi*'(u) = log(u)
    if u <= 0: return np.nan # Domaine non valide
    return np.log(u)
    
  if np.isclose(np.abs(alpha), np.inf): # Cas Psi_inf: Psi*'(u) = -1/u
    if u >= 0: return np.nan # Domaine non valide
    return -1.0 / u
  
  # Cas Généraux: |alpha|<1 ou |alpha|>1
  
  q = get_q(alpha)
  
  if np.abs(alpha) < 1.0: # |alpha|<1 : Psi*'(u) = u^(q-1)
    return np.power(u, q - 1.0)
    
  elif np.abs(alpha) > 1.0: # |alpha|>1 : Psi*'(u) = (-u)^(q-1)
    if u >= 0: return np.nan # Domaine non valide (nécessite u < 0)
    return np.power(-u, q - 1.0)
    
  return np.nan # Cas non géré

# --- Fonction de Recherche de Racine ---

# NOTE: Pour une implémentation Numba complète, ce solveur doit être implémenté 
# avec une méthode Numba-compatible (ex: sécante ou Newton).
def func_to_solve(gamma, s_block, alpha):
  """Fonction dont on cherche la racine: sum(Psi*'(s_i - gamma)) = 0."""
  u_block = s_block - gamma
  sum_of_gradients = 0.0
  for u in u_block:
    grad = psi_star_prime(u, alpha)
    if np.isnan(grad):
      # Gestion des domaines (si gamma est en dehors du domaine valide)
      # Pour un solveur, on retourne une valeur qui pousse gamma dans le bon sens
      return np.inf 
    sum_of_gradients += grad
  return sum_of_gradients


# --- Algorithme PAV Généralisé ---

@njit
def isotonic_general_pav(s, w, alpha, sol):
  """Résout l'optimisation isotonique avec régularisation générale (alpha-divergence).
  
  Args:
    s: input de la régression, un 1d-array.
    w: input de pondération (nécessaire pour le cas alpha=1).
    alpha: paramètre de la divergence.
    sol: où écrire la solution (v), un array de même taille que s.
  """
  n = s.shape[0]
  target = np.arange(n)
  
  # Pour le cas alpha=1 (Entropique), nous devons suivre l'agrégation LSE.
  # Les cas généraux (non fermés) nécessitent uniquement 's'.
  
  # Initialisation: sol[i] = s[i] - w[i] (pour être cohérent avec KL/Entropique)
  for i in range(n):
    # Pour L2 (alpha=0), w n'est pas utilisé et sol[i]=s[i]. Nous le gérons plus tard.
    sol[i] = max(1, s[i] - w[i])
    
  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
      
    # Test de violation:
    if sol[i] > sol[k]:
      i = k
      continue
      
    # Violation: Fusionner les blocs [i...target[i]] et [k...target[k]]
    
    block_start = i
    block_end = target[k]
    
    # 1. Calcul de la valeur agrégée gamma
    if np.isclose(alpha, 0): # Cas L2 (forme fermée)
      s_block = s[block_start : block_end + 1]
      gamma = np.mean(s_block)
    
    elif np.isclose(alpha, 1): # Cas Entropique (forme fermée avec LSE)
      # Pour cette divergence, la solution agrégée est: LSE(s) - LSE(w)
      s_block = s[block_start : block_end + 1]
      w_block = w[block_start : block_end + 1]
      # Utiliser LogSumExp pour la stabilité numérique:
      log_sum_exp_s = np.log(np.sum(np.exp(s_block)))
      log_sum_exp_w = np.log(np.sum(np.exp(w_block)))
      gamma = log_sum_exp_s - log_sum_exp_w
      
    else: # Cas Général (requiert solveur numérique)
      # *Cette partie doit être remplacée par un solveur de racine Numba-compatible*
      s_block = s[block_start : block_end + 1]
      
      # TODO: Définir une plage [a, b] et utiliser un solveur comme la sécante:
      # Ex: gamma = solve_root(func_to_solve, a, b, s_block, alpha)
      gamma = s[i] # Placeholder: utiliser la valeur de s pour le moment, 
                   # car le solveur externe n'est pas intégré.
      
    # 2. Mise à jour du bloc agrégé
    sol[i] = gamma
    
    # 3. Mise à jour des pointeurs
    target[i] = block_end
    target[block_end] = i
    
    # 4. Backtracking
    if i > 0:
      i = target[i - 1]
    # Sinon, on recommence au début du nouveau bloc
    
  # --- Reconstruction de la solution finale ---
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k