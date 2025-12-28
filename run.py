import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import copy 
from simul import Simul
from animate import Animate

def get_lyapunov_slope(time_array: np.ndarray, distances: list, width_threshold: float) -> float:
    """
    Estime l'exposant de Lyapunov par régression linéaire sur le log de la distance.
    Filtre automatiquement la phase de saturation.

    IN:
        time_array (np.ndarray): Tableau des temps.
        distances (list): Liste des distances euclidiennes entre les deux particules.
        width_threshold (float): Largeur du stade (critère de saturation).
    OUT:
        slope (float): Pente de la régression (l'exposant Lambda).
    """
    safe_dist = np.array(distances)
    # Remplacement des 0 par epsilon pour éviter log(0) = -inf
    safe_dist[safe_dist == 0] = 1e-10
    log_dist = np.log(safe_dist)
    
    # Seuil de coupure : on arrête l'analyse quand la distance atteint le rayon (W/2)
    threshold = np.log(width_threshold / 2.0)
    
    # Création du masque pour la phase transitoire (avant saturation)
    mask = log_dist < threshold
    if np.any(~mask): 
        first_sat = np.argmax(~mask)
        # Si la saturation arrive trop vite (<10 points), on ignore
        if first_sat > 10: mask[first_sat:] = False
        else: return 0.0
            
    t_fit = time_array[mask]
    y_fit = log_dist[mask]
    
    # Sécurité statistique
    if len(t_fit) < 10: return 0.0
    
    # Régression linéaire y = ax + b -> a est l'exposant de Lyapunov
    slope, _ = np.polyfit(t_fit, y_fit, 1)
    return slope


def generate_heatmap(x_data: list, y_data: list, L: float, W: float) -> None:
    """
    Génère et affiche une carte de chaleur des positions visitées.
    Utilise un masquage géométrique pour épouser la forme du stade.

    IN:
        x_data (list): Historique des positions X.
        y_data (list): Historique des positions Y.
        L (float): Longueur du stade.
        W (float): Largeur du stade.
    OUT:
        None (Affiche la figure Matplotlib).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white') 
    ax.set_facecolor('white')

    # Calcul de la résolution de la grille (cellules carrées)
    aspect_ratio = (L + W) / W if W > 0 else 1
    ny = 100  # Résolution verticale
    nx = int(ny * aspect_ratio) 
    
    # Définition des bornes de l'histogramme avec marge
    margin = W * 0.05
    x_edges = np.linspace(-W/2 - margin, L + W/2 + margin, nx + 1)
    y_edges = np.linspace(-margin, W + margin, ny + 1)

    # Calcul de l'histogramme 2D (Densité de probabilité)
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=(x_edges, y_edges), density=True)
    H = H.T # Transposition nécessaire pour pcolormesh

    # --- Création du Masque Géométrique ---
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    XC, YC = np.meshgrid(xc, yc)

    R = W / 2 
    # Zones valides : Rectangle central + Demi-cercle gauche + Demi-cercle droit
    mask_rect = (XC >= 0) & (XC <= L) & (YC >= 0) & (YC <= W)
    dist_left = np.sqrt((XC - 0)**2 + (YC - R)**2)
    mask_left = (XC < 0) & (dist_left <= R)
    dist_right = np.sqrt((XC - L)**2 + (YC - R)**2)
    mask_right = (XC > L) & (dist_right <= R)

    stadium_mask = mask_rect | mask_left | mask_right
    # Application du masque (les zones hors stade deviennent NaN)
    H_masked = np.where(stadium_mask, H, np.nan)

    # Gestion de la colormap pour rendre les NaNs transparents
    cmap = copy.copy(plt.get_cmap('inferno'))
    cmap.set_bad(color='white', alpha=0) 

    mesh = ax.pcolormesh(xedges, yedges, H_masked, cmap=cmap, shading='auto')
    
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Densité de probabilité de présence")
    
    # --- Tracé des bordures du stade ---
    def plot_double_line(x, y):
        """Trace une ligne noire épaisse surmontée d'une ligne blanche fine."""
        ax.plot(x, y, 'k-', lw=5, zorder=10)
        ax.plot(x, y, 'w-', lw=3, zorder=11)

    # Murs plats
    plot_double_line([0, L], [0, 0]) 
    plot_double_line([0, L], [W, W]) 
    
    # Arcs
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, 200)
    plot_double_line(0 + R*np.cos(theta_left), R + R*np.sin(theta_left))
    
    theta_right = np.linspace(-np.pi/2, np.pi/2, 200)
    plot_double_line(L + R*np.cos(theta_right), R + R*np.sin(theta_right))
    
    ax.set_title(f"Carte de chaleur (L={L}, W={W})")
    ax.set_aspect('equal')
    ax.set_xlim(-W/2 - margin, L + W/2 + margin)
    ax.set_ylim(-margin, W + margin)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_chaos_single(time_array: np.ndarray, distances: list, W: float) -> None:
    """
    Affiche l'analyse de divergence pour une simulation unique :
    1. Courbe de distance linéaire.
    2. Courbe semi-logarithmique avec régression de Lyapunov.

    IN:
        time_array (np.ndarray): Axe temporel.
        distances (list): Distances mesurées.
        W (float): Largeur du stade (pour le calcul du seuil).
    OUT:
        None.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphe 1 : Distance linéaire
    ax1.plot(time_array, distances, color='purple', alpha=0.9)
    ax1.set_title("Distance Euclidienne $d(t)$")
    ax1.set_xlabel("Temps (u.a)")
    ax1.set_ylabel("$d(t)$ (u.a)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Graphe 2 : Logarithme
    safe_dist = np.array(distances)
    safe_dist[safe_dist == 0] = 1e-10
    log_dist = np.log(safe_dist)
    
    ax2.plot(time_array, log_dist, color='#5DADE2', alpha=0.5, label='Données brutes')
    
    # Calcul de la pente
    slope = get_lyapunov_slope(time_array, distances, W)
    
    # Affichage de la droite de régression sur la zone pertinente
    threshold = np.log(W / 2.0)
    mask = log_dist < threshold
    if np.any(~mask): mask[np.argmax(~mask):] = False
    
    if np.sum(mask) > 5:
        t_fit = time_array[mask]
        # Recalcul de l'ordonnée à l'origine pour l'affichage visuel
        reg = slope * t_fit + (np.mean(log_dist[mask]) - slope * np.mean(t_fit))
        ax2.plot(t_fit, reg, 'r--', lw=2, label=f"Régression (λ={slope:.3f})")
    
    ax2.set_title(f"Exposant de Lyapunov : $\lambda \\approx {slope:.3f}$")
    ax2.set_xlabel("Temps (u.a)")
    ax2.set_ylabel("$ln(d(t))$")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    """
    Point d'entrée principal du programme. Affiche le menu interactif 
    et lance les simulations selon le choix de l'utilisateur.
    """
    W = 6.0         # Largeur fixe du stade
    sigma = 0.15    # Rayon des particules
    dt = 0.05       # Pas de temps
    
    print("\n====================== Simulation Stade de Bunimovich ======================\n")

    print("1. Une particule (Simulation ou Heatmap)")
    print("2. Deux particules (Simulation ou Analyse de la divergence)")
    print("3. Deux particules avec collisions (Simulation ou Analyse de la divergence)")
    print("4. Etude de la divergence en fonction de la taille du stade : λ = f(L)")
    
    choix = input("\n Votre choix (1/2/3/4) : ")
    
    # --- CAS 4 : ETUDE PARAMETRIQUE COMPLETE ---
    if choix == '4':
        print("\n Lancement de l'étude paramétrique...")
        
        # Construction d'un échantillonnage intelligent de L (Longueur)
        # Zone de transition fine (logspace) + Zone standard + Zone asymptotique
        L_transition = np.logspace(-3, 0, 25) 
        L_standard = np.linspace(1.5, 20, 20)
        L_long = np.geomspace(25, 200, 15)
        L_values = np.concatenate(([0], L_transition, L_standard, L_long))
        L_values = np.sort(np.unique(L_values))
        
        print(f" Calcul sur {len(L_values)} simulations...")
        lyapunov_values = []
        nframes_study = 2000 
        
        # Boucle sur les différentes géométries
        for i, val_L in enumerate(L_values):
            sim = Simul(dt, sigma, L=val_L, W=W, N=2, interactions=False)
            dists = []
            for _ in range(nframes_study):
                dists.append(sim.md_step())
            
            times = np.linspace(0, nframes_study*dt, nframes_study)
            lam = get_lyapunov_slope(times, dists, width_threshold=W)
            
            if lam < 0: lam = 0 
            lyapunov_values.append(lam)
            
            # Barre de chargement
            progression = int((i+1)/len(L_values) * 20)
            print(f"[{'#'*progression}{'-'*(20-progression)}] L={val_L:6.2f} -> λ={lam:.3f}")
            
        # Affichage des résultats (Double échelle Lin/Log)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(L_values, lyapunov_values, 'o-', color='darkgreen', lw=1.5, markersize=4)
        ax1.set_xlabel("L (Linéaire)"); ax1.set_ylabel("$\lambda$"); ax1.set_title("Vue d'ensemble")
        ax1.grid(True, alpha=0.5)
        ax1.axvline(0, color='k', lw=1)
        
        ax2.plot(L_values, lyapunov_values, 'o-', color='darkblue', lw=1.5, markersize=4)
        ax2.set_xscale('log'); ax2.set_xlabel("L (Log)"); ax2.set_ylabel("$\lambda$"); ax2.set_title("Zoom Transition")
        ax2.grid(True, which="both", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # --- CAS 1, 2, 3 : SIMULATIONS UNIQUES ---
    else:
        L_default = 10 # Longueur par défaut pour les démos
        
        if choix == '1':
            visu = input("\n Visualiser l'animation (o) ? Ou visualiser l'analyse de l'ergodicité (n) ? (o/n) : ").lower() == 'o'
            sim = Simul(dt, sigma, L=L_default, W=W, N=1)
            
            if visu:
                # Mode Animation
                app = Animate(sim)
                app.start(1500)
                print("")
            else:
                # Mode Heatmap (Ergodicité)
                x_pos = []
                y_pos = []
                nframes_heatmap = 100000 
                
                print(f"\n Calcul de la Heatmap (L={L_default})...")
                print(f" Simulation longue ({nframes_heatmap} pas) pour un remplissage optimal...\n")
                
                for _ in range(nframes_heatmap):
                    sim.md_step()
                    x_pos.append(sim.position[0, 0])
                    y_pos.append(sim.position[0, 1])
                
                generate_heatmap(x_pos, y_pos, L_default, W)

        elif choix in ['2', '3']:
            interact = (choix == '3') # Active les collisions inter-particules pour 3
            sim = Simul(dt, sigma, L=L_default, W=W, N=2, interactions=interact)
            visu = input("\n Visualiser l'animation (o) ? Ou visualiser l'analyse de la divergence (n) ? (o/n) : ").lower() == 'o'
            
            if visu:
                app = Animate(sim)
                app.start(1500)
                print("\n Pour voir les courbes d'analyse, relancez sans l'animation.\n")
            else:
                print("\n Calcul de Lyapunov en cours...\n")
                distances = []
                nframes = 1500
                temps = np.linspace(0, nframes*dt, nframes)
                for _ in range(nframes):
                    distances.append(sim.md_step())
                analyze_chaos_single(temps, distances, W)

if __name__ == '__main__':
    main()