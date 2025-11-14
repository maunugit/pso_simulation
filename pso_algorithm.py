import numpy as np
from typing import Callable


class Particle:
    """
    Represents a single particle in the swarm, flying around in the 2D space
    """
    
    def __init__(self, bounds, dim=2):
        """
        Initializes particle with random position and velocity
        
        Args:
            bounds: tuple (min, max) defining search space
            dim: int, dimensionality (default 2 for 2D visualization)
        """
        self.dim = dim
        self.bounds = bounds
        
        # initialize random position within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        
        # initialize random velocity
        velocity_range = (bounds[1] - bounds[0]) * 0.1 # 10% of the full span to avoid huge jumps
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dim)
        
        # personal best
        self.best_position = self.position.copy() # always starts at the initial pos
        self.best_fitness = float('inf') # starts as 'inf', will be updated on first eval
        
        # current fitness
        self.fitness = float('inf')
    
    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Updates particle velocity using the PSO formula
        
        v = w*v + c1*r1*(personal_best - position) + c2*r2*(global_best - position)
        """
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        # inertia component
        inertia = w * self.velocity
        
        # cognitive component (personal best)
        cognitive = c1 * r1 * (self.best_position - self.position)
        
        # social component (global best)
        social = c2 * r2 * (global_best_position - self.position)
        
        # update velocity, moves the particle by adding velocity to the pos
        # also keeps it in the bounds
        self.velocity = inertia + cognitive + social
        # velocity can be limited to prevent particles from flying too fast
        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.2 # 20% of total range
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """
        Updates particle position based on velocity
        """
        self.position = self.position + self.velocity
        
        # keep particle within bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    
    def evaluate(self, objective_function: Callable):
        """
        Computes the fitness at the current position. Updates best-position and best-fitness
        
        Args:
            objective_function: which function to minimize
        """
        self.fitness = objective_function(self.position)
        
        # update personal best if current position is better
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class PSO:
    """
    Particle Swarm Optimization algorithm
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 bounds: tuple,
                 num_particles: int = 30,
                 num_iterations: int = 200,
                 w: float = 0.7, # inertia weight (keeps the particle moving)
                 c1: float = 1.5, # cognitive factor (how strongly it follows its own PB)
                 c2: float = 1.5, # social factor (how strongly it follows the swarm)
                 dim: int = 2):
        """
        Initialize PSO optimizer
        
        Args:
            objective_function: function to minimize
            bounds: tuple (min, max) defining search space
            num_particles: number of particles in swarm
            num_iterations: maximum number of iterations
            w: inertia weight
            c1: cognitive coefficient (personal best attraction)
            c2: social coefficient (global best attraction)
            dim: dimensionality (2 for 2D problems)
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = dim
        
        # initialize swarm
        self.particles = [Particle(bounds, dim) for _ in range(num_particles)]
        
        # global best
        self.global_best_position = None

        # global best fitness, lower function value = better fitness
        """
        If a particle is at a spot in the landscape where either function outputs
        a small number, that particle is considered "fit". If it's a large num, it's unfit.
        """
        self.global_best_fitness = float('inf')
        
        # history tracking for analysis
        self.fitness_history = []  # Track best fitness per iteration
        self.position_history = []  # Track all particle positions per iteration
        
        # current iteration
        self.iteration = 0
        
        # initialize by evaluating all particles
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """
        Evaluate initial positions and set global best
        """
        for particle in self.particles:
            particle.evaluate(self.objective_function)
            
            # Update global best
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # record initial state
        self._record_state()
    
    def _record_state(self):
        """
        Record current state for history tracking
        """
        self.fitness_history.append(self.global_best_fitness)
        
        # Record all particle positions for animation
        positions = np.array([p.position for p in self.particles])
        self.position_history.append(positions.copy())
    
    def step(self):
        """
        Perform one iteration of PSO algorithm
        1. Updates velocity
        2. Updates position
        3. Evaluates fitness
        4. If the particle now has better fitness than the current global best, update global best
        5. Record states and increment
        Returns:
            float: current global best fitness
        """
        # update all particles
        for particle in self.particles:
            # update velocity
            particle.update_velocity(self.global_best_position, 
                                    self.w, self.c1, self.c2)
            
            # update position
            particle.update_position()
            
            # evaluate new position
            particle.evaluate(self.objective_function)
            
            # update global best if necessary
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # record state
        self._record_state()
        
        # increment iteration counter
        self.iteration += 1
        
        return self.global_best_fitness
    
    def optimize(self, verbose=True):
        """
        Run the complete optimization process
        
        Args:
            verbose: bool, whether to print progress
        
        Returns:
            tuple: (best_position, best_fitness)
        """
        if verbose:
            print(f"Starting PSO optimization...")
            print(f"Particles: {self.num_particles}, Iterations: {self.num_iterations}")
            print(f"Parameters: w={self.w}, c1={self.c1}, c2={self.c2}")
            print("-" * 60)
        
        for i in range(self.num_iterations):
            fitness = self.step()
            
            if verbose and (i + 1) % 20 == 0:
                print(f"Iteration {i + 1}/{self.num_iterations}: "
                      f"Best Fitness = {fitness:.6f}")
        
        if verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            print(f"Best position: {self.global_best_position}")
            print(f"Best fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best_position, self.global_best_fitness
    
    def get_particle_positions(self):
        """
        Get current positions of all particles
        
        Returns:
            numpy array of shape (num_particles, dim)
        """
        return np.array([p.position for p in self.particles])
    
    def get_fitness_history(self):
        """
        Get history of best fitness values
        
        Returns:
            list of fitness values
        """
        return self.fitness_history
    
    def get_position_history(self):
        """
        Get history of all particle positions
        
        Returns:
            list of numpy arrays, each of shape (num_particles, dim)
        """
        return self.position_history
    
    def reset(self):
        """
        Reset the optimizer to initial state
        """
        self.particles = [Particle(self.bounds, self.dim) 
                         for _ in range(self.num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        self.position_history = []
        self.iteration = 0
        self._initialize_swarm()


# test the PSO implementation
if __name__ == "__main__":
    from objective_functions import schwefel, rosenbrock
    
    print("=" * 60)
    print("Testing PSO on Schwefel Function")
    print("=" * 60)
    
    pso_schwefel = PSO(
        objective_function=schwefel,
        bounds=(-500, 500),
        num_particles=50,        
        num_iterations=300,      
        w=0.5,                   
        c1=2.0,                  
        c2=2.0                   
    )
    
    best_pos, best_fit = pso_schwefel.optimize(verbose=True)
    print(f"\nKnown optimum: [420.9687, 420.9687]")
    print(f"Found optimum: {best_pos}")
    print(f"Error: {np.linalg.norm(best_pos - np.array([420.9687, 420.9687])):.4f}")
    
    print("\n" + "=" * 60)
    print("Testing PSO on Rosenbrock Function")
    print("=" * 60)
    
    pso_rosenbrock = PSO(
        objective_function=rosenbrock,
        bounds=(-2, 2),
        num_particles=30,
        num_iterations=100,
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    best_pos, best_fit = pso_rosenbrock.optimize(verbose=True)
    print(f"\nKnown optimum: [1.0, 1.0]")
    print(f"Found optimum: {best_pos}")
    print(f"Error: {np.linalg.norm(best_pos - np.array([1.0, 1.0])):.4f}")