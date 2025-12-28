import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.patches import Arc
import numpy as np
from simul import Simul # Type hinting

class Animate:
    """
    Classe gérant l'animation graphique du stade de Bunimovich.
    Utilise Matplotlib pour un rendu temps réel des particules et trajectoires.
    """
    
    def __init__(self, simulation: Simul):
        """
        Initialise la figure, les axes et les objets graphiques statiques (murs).

        IN:
            simulation (Simul): Instance de la classe Simul contenant l'état physique.
        OUT:
            None
        """
        self.sim = simulation
        
        # Adaptation de la taille de la fenêtre graphique selon le ratio L/W
        fig_w = 6 + simulation.L / 2
        self.fig, self.ax = plt.subplots(figsize=(min(12, fig_w), 6))
        
        # --- Dessin Statique du Stade ---
        # Lignes horizontales (haut et bas)
        walls = [[[0, 0], [simulation.L, 0]], 
                 [[0, simulation.W], [simulation.L, simulation.W]]]
        self.ax.add_collection(LineCollection(walls, colors='black', linewidths=2))
        
        # Arcs de cercle (gauche et droite)
        # theta1/2 définissent les angles de l'arc (270->90 droite, 90->270 gauche)
        self.ax.add_patch(Arc((simulation.L, simulation.W/2), simulation.W, simulation.W, 
                              theta1=270, theta2=90, lw=2))
        self.ax.add_patch(Arc((0, simulation.W/2), simulation.W, simulation.W, 
                              theta1=90, theta2=270, lw=2))
        
        # --- Objets Dynamiques ---
        # Utilisation d'EllipseCollection pour traiter N particules
        colors = ['blue', 'red'] if simulation.N > 1 else ['blue']
        self.particles = EllipseCollection(
            widths=2*simulation.sigma, heights=2*simulation.sigma, angles=0, units='x',
            offsets=simulation.position, transOffset=self.ax.transData, facecolors=colors
        )
        self.ax.add_collection(self.particles)
        
        # Initialisation des traînées pour visualiser la trajectoire
        self.trails = [self.ax.plot([], [], linestyle=':', color=c, alpha=0.6, lw=1)[0] for c in colors]
        self.history_x = [[] for _ in range(simulation.N)]
        self.history_y = [[] for _ in range(simulation.N)]

        # --- Configuration des Axes ---
        margin = simulation.W * 0.2
        self.ax.set_xlim(-simulation.W/2 - margin, simulation.L + simulation.W/2 + margin)
        self.ax.set_ylim(-margin, simulation.W + margin)
        self.ax.set_aspect('equal')
        self.ax.axis('off') # Suppression des axes numériques pour l'esthétique


    def update(self, frame: int) -> tuple:
        """
        Boucle de mise à jour appelée à chaque image de l'animation.
        Avance la physique et met à jour les graphiques.

        IN:
            frame (int): Numéro de la frame courante (fourni par FuncAnimation).
        OUT:
            artists (tuple): Liste des objets graphiques modifiés (pour le blitting).
        """
        # Avance la physique d'un pas
        self.sim.md_step()
        
        # Mise à jour des positions des particules
        self.particles.set_offsets(self.sim.position)
        
        # Mise à jour des traînées (historique complet)
        for i in range(self.sim.N):
            self.history_x[i].append(self.sim.position[i, 0])
            self.history_y[i].append(self.sim.position[i, 1])
            self.trails[i].set_data(self.history_x[i], self.history_y[i])
            
        return self.particles, *self.trails


    def start(self, nframes: int) -> None:
        """
        Lance l'animation.

        IN:
            nframes (int): Nombre total d'images à calculer.
        OUT:
            None
        """
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=nframes, interval=20, blit=False)
        plt.show()