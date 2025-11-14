"""
Configuration parameters for PSO simulation
"""

# PSO Parameters
NUM_PARTICLES = 30 # 30, 50
NUM_ITERATIONS = 100 # 100, 150, 200
INERTIA_WEIGHT = 0.7 # w: controls momentum / keeps the particle moving
COGNITIVE_COEFF = 1.5 # c1: attraction to personal best / how strongly it follows its PB
SOCIAL_COEFF = 1.5 # c2: attraction to global best / how strongly it follows the swarm

# schwefel Function Parameters
SCHWEFEL_BOUNDS = (-500, 500)
SCHWEFEL_DIM = 2

# rosenbrock Function Parameters
ROSENBROCK_BOUNDS = (-2, 2)
ROSENBROCK_DIM = 2

# animation Parameters
ANIMATION_INTERVAL = 50
DPI = 100
FPS = 20 

# visualization Parameters
CONTOUR_LEVELS = 30 
PARTICLE_SIZE = 50
PARTICLE_COLOR = 'red'
BEST_PARTICLE_COLOR = 'yellow'
GRID_RESOLUTION = 100