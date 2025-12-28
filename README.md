# Bunimovich Stadium – Chaos Simulation

This project is a Python implementation of the Bunimovich Stadium billiard (a rectangle capped by semicircles). It simulates the chaotic dynamics of particles using an event-driven algorithm to analyze divergence, ergodicity, and the transition to chaos.

## Features

* **Event-Driven Physics:** Exact calculation of collision times for walls and arcs (no fixed time-step error).
* **Chaos Quantification:** Computation of the Lyapunov exponent ($\lambda$) via linear regression on trajectory divergence.
* **Ergodicity Analysis:** Generation of high-resolution Heatmaps to visualize phase space exploration.
* **Visualization:** Real-time animation of particles and trajectory trails.
* **Multi-Particle Dynamics:** Simulation of two particles with or without elastic inter-particle collisions.
* **Parametric Study:** Automated analysis of the transition from integrable (Circle) to chaotic (Stadium) regimes.

## Requirements

* numpy
* matplotlib

Install them via:
```bash
pip install -r requirements.txt
