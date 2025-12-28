import numpy as np
import math

class Simul:
    """
    Moteur de simulation de dynamique moléculaire dans un stade de Bunimovich.
    Gère la cinématique des particules, la détection des collisions et la 
    résolution des chocs élastiques.
    """

    def __init__(self, simul_time: float, sigma: float, L: float, W: float, N: int, interactions: bool = False):
        """
        Initialise l'état de la simulation (positions, vitesses, géométrie).

        IN:
            simul_time (float): Pas de temps maximal d'intégration par étape (dt).
            sigma (float): Rayon des particules.
            L (float): Longueur des parois plates horizontales (Axe X).
            W (float): Largeur/Hauteur du stade (Axe Y) = Diamètre des cercles.
            N (int): Nombre de particules (1 ou 2).
            interactions (bool): Active les collisions entre particules si True.
        OUT:
            None
        """
        self.L = L  
        self.W = W  
        self.sigma = sigma
        self.simul_time = simul_time
        self.interactions = interactions
        self.N = N

        # Initialisation des tableaux d'état (Position et Vitesse)
        self.position = np.zeros((N, 2))
        self.velocity = np.zeros((N, 2))
        
        # Positionnement de la particule 1 au centre du stade
        self.position[0] = [L/2, W/2]
        self.velocity[0] = [3.5, 2.0] # Vitesse arbitraire non-nulle
        
        # Si N=2, on place la seconde particule à une distance infinitésimale 
        # pour étudier la divergence des trajectoires (Chaos)
        if N > 1:
            epsilon = 1e-5
            self.position[1] = self.position[0] + [0, epsilon]
            self.velocity[1] = self.velocity[0] + [epsilon, 0]


    def wall_time(self) :
        """
        Calcule le temps avant la prochaine collision avec les murs horizontaux (plats).
        
        IN:
            None (Utilise l'état interne self.position, self.velocity).
        OUT:
            min_dt (float): Temps minimum avant collision.
            idx (int): Indice de la particule concernée.
        """
        # Calcul vectoriel : distance vers le mur du haut (W) ou du bas (0) 
        # selon le signe de Vy
        dist_y = np.where(self.velocity[:, 1] > 0,
                          self.W - self.sigma - self.position[:, 1], # Vers le haut
                          self.position[:, 1] - self.sigma)          # Vers le bas
        
        # Gestion des erreurs de division par zéro (si Vy = 0)
        with np.errstate(divide='ignore'):
            dt = dist_y / np.abs(self.velocity[:, 1])

        # Projection : où sera la particule en X au moment de l'impact théorique ?
        x_impact = self.position[:, 0] + dt * self.velocity[:, 0]

        # Validité : Le temps doit être positif ET l'impact doit se faire 
        # sur la partie plate (entre x=0 et x=L).
        valid = (dt > 1e-9) & (x_impact >= 0) & (x_impact <= self.L)
        
        # On remplace les temps invalides par l'infini
        dt_final = np.where(valid, dt, np.inf)

        idx = np.argmin(dt_final)
        return dt_final[idx], idx


    def arc_time(self, center_x: float, right: bool = True) :
        """
        Calcule le temps de collision avec un arc de cercle (gauche ou droit).
        Résout l'équation quadratique d'intersection ligne-cercle.

        IN:
            center_x (float): Coordonnée X du centre de l'arc (0 ou L).
            right (bool): True si c'est l'arc de droite, False pour gauche.
        OUT:
            min_dt (float): Temps minimum avant collision.
            idx (int): Indice de la particule.
        """
        center = np.array([center_x, self.W / 2])
        delta = self.position - center
        v = self.velocity
        
        # Rayon effectif = Rayon stade - Rayon particule
        R_eff = (self.W / 2) - self.sigma 
        
        # Coefficients de l'équation quadratique at² + bt + c = 0
        a = np.sum(v**2, axis=1)
        b = 2 * np.sum(delta * v, axis=1)
        c = np.sum(delta**2, axis=1) - R_eff**2
        
        delta_q = b**2 - 4 * a * c
        
        # Utilisation de errstate pour ignorer les warnings numpy (sqrt de négatif)
        # pour les particules qui ne se dirigent pas vers le cercle.
        with np.errstate(invalid='ignore'):
            # On cherche la solution t = (-b + sqrt(delta)) / 2a
            # C'est la sortie du cercle mathématique, donc le choc intérieur
            t = (-b + np.sqrt(delta_q)) / (2 * a)

            # Filtre géométrique : vérifier que l'impact est du bon côté
            if right:
                condition = (self.position[:, 0] + t * self.velocity[:, 0]) >= self.L
            else:
                condition = (self.position[:, 0] + t * self.velocity[:, 0]) <= 0
                
            # Validité globale : discriminant positif, temps futur, et bonne géométrie
            t_final = np.where((delta_q >= 0) & (t > 1e-9) & condition, t, np.inf)
            
        idx = np.argmin(t_final)
        return t_final[idx], idx


    def particle_collision_time(self) :
        """
        Calcule le temps avant collision entre deux particules (si activé).
        
        IN:
            None.
        OUT:
            dt (float): Temps avant collision.
            pair (tuple): Indices des particules (0, 1) ou None.
        """
        if not self.interactions or self.N < 2: 
            return np.inf, None
            
        dp = self.position[0] - self.position[1]
        dv = self.velocity[0] - self.velocity[1]
        
        # Coefficients quadratiques pour la distance relative
        a = np.dot(dv, dv)
        b = 2 * np.dot(dp, dv)
        c = np.dot(dp, dp) - (2 * self.sigma)**2 
        
        delta = b**2 - 4*a*c
        
        if delta < 0: 
            return np.inf, None
            
        # Solution d'entrée en collision (la plus petite racine)
        t = (-b - np.sqrt(delta)) / (2*a)
        
        return (t, (0, 1)) if t > 1e-9 else (np.inf, None)


    def resolve_collisions(self, idx, type_col: str, center_x: float = None, pair: tuple = None) -> None:
        """
        Applique les lois de réflexion (élastique) sur les vitesses après un choc.

        IN:
            idx (int): Indice de la particule (None si collision entre particules).
            type_col (str): Type de collision ('wall', 'arc', 'part').
            center_x (float): Centre X de l'arc (si type_col='arc').
            pair (tuple): Paires d'indices (si type_col='part').
        OUT:
            None (Modifie self.velocity in-place).
        """
        if type_col == 'wall':
            # Rebond sur mur horizontal : inversion de la composante Y
            self.velocity[idx, 1] *= -1
        
        elif type_col == 'arc':
            # Rebond sur cercle : réflexion spéculaire par rapport à la normale
            center = np.array([center_x, self.W / 2])
            normal = self.position[idx] - center
            normal /= np.linalg.norm(normal)
            
            # Formule v_new = v - 2(v.n)n
            v_dot_n = np.dot(self.velocity[idx], normal)
            self.velocity[idx] -= 2 * v_dot_n * normal
            
        elif type_col == 'part':
            # Choc élastique entre deux sphères dures de même masse
            r1, r2 = self.position[0], self.position[1]
            v1, v2 = self.velocity[0], self.velocity[1]
            
            dist_sq = np.sum((r1-r2)**2)
            dot = np.dot(v1-v2, r1-r2)
            
            # Échange d'impulsion le long de l'axe de collision
            self.velocity[0] = v1 - (dot/dist_sq)*(r1-r2)
            self.velocity[1] = v2 - (dot/dist_sq)*(r2-r1)


    def md_step(self) -> float:
        """
        Avance la simulation d'un pas de temps complet (simul_time) en gérant
        les collisions intermédiaires.

        IN:
            None.
        OUT:
            distance (float): Distance euclidienne entre les particules à chaque pas.
        """
        t_left = self.simul_time
        
        while t_left > 0:
            # Calcul de tous les temps de collision possibles
            t_w, i_w = self.wall_time()
            t_r, i_r = self.arc_time(self.L, right=True)   # Arc Droit
            t_l, i_l = self.arc_time(0, right=False)       # Arc Gauche
            t_p, pair = self.particle_collision_time()
            
            # Identification de l'événement le plus proche
            min_dt = min(t_w, t_r, t_l, t_p)
            
            # Si aucun événement avant la fin du pas, on avance simplement
            if min_dt > t_left:
                self.position += self.velocity * t_left
                break
            
            # Sinon, on avance jusqu'à la collision et on la résout
            self.position += self.velocity * min_dt
            t_left -= min_dt
            
            if min_dt == t_w: self.resolve_collisions(i_w, 'wall')
            elif min_dt == t_r: self.resolve_collisions(i_r, 'arc', center_x=self.L)
            elif min_dt == t_l: self.resolve_collisions(i_l, 'arc', center_x=0)
            elif min_dt == t_p: self.resolve_collisions(None, 'part', pair=pair)

        # Retour de la métrique de distance pour l'analyse
        if self.N > 1:
            return np.linalg.norm(self.position[0] - self.position[1])
        return 0.0