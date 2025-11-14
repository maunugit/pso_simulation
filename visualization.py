import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm
import os


class PSOVisualizer:
    
    def __init__(self, pso, function_name, output_dir='outputs'):
        """
        Initialize visualizer
        
        Args:
            pso: PSO object with optimization results
            function_name: str, name of the function being optimized
            output_dir: str, directory to save outputs
        """
        self.pso = pso
        self.function_name = function_name
        self.output_dir = output_dir
        
        # create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # function bounds
        self.bounds = pso.bounds
        
        # mesh grid for contour plotting
        self.resolution = 100
        self.X, self.Y = self._create_mesh()
        self.Z = self._evaluate_mesh()
    
    def _create_mesh(self):
        """Create meshgrid for contour plotting"""
        x = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        return np.meshgrid(x, y)
    
    def _evaluate_mesh(self):
        """Evaluate objective function on mesh grid"""
        Z = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                position = np.array([self.X[i, j], self.Y[i, j]])
                Z[i, j] = self.pso.objective_function(position)
        return Z
    
    def plot_convergence(self, save=True):
        """
        Plot convergence curve showing fitness improvement over iterations
        
        Args:
            save: bool, whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fitness_history = self.pso.get_fitness_history()
        iterations = range(len(fitness_history))
        
        ax.plot(iterations, fitness_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title(f'PSO Convergence - {self.function_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(fitness_history))
        
        # add final value annotation
        final_fitness = fitness_history[-1]
        ax.annotate(f'Final: {final_fitness:.6f}',
                   xy=(len(fitness_history)-1, final_fitness),
                   xytext=(-60, 20),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'plots', 
                                   f'{self.function_name}_convergence.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved: {filename}")
        
        plt.show()
        return fig
    
    def plot_landscape(self, show_optimum=True, save=True):
        """
        Plot the objective function landscape
        
        Args:
            show_optimum: bool, whether to mark the known optimum
            save: bool, whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # create contour plot
        contour = ax.contour(self.X, self.Y, self.Z, levels=30, # not using this
                            cmap='viridis', alpha=0.6)
        contourf = ax.contourf(self.X, self.Y, self.Z, levels=30, 
                              cmap='viridis', alpha=0.4)
        
        # add colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Function Value', fontsize=12)
        
        # mark optimum if requested
        if show_optimum:
            if 'schwefel' in self.function_name.lower():
                optimum = [420.9687, 420.9687]
            else:  # rosenbrock
                optimum = [1.0, 1.0]
            ax.plot(optimum[0], optimum[1], 'r*', markersize=20, 
                   label='Global Optimum', markeredgecolor='white', markeredgewidth=1.5)
        
        # mark PSO result
        best_pos = self.pso.global_best_position
        ax.plot(best_pos[0], best_pos[1], 'yo', markersize=15, 
               label='PSO Result', markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'{self.function_name} Landscape', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'plots', 
                                   f'{self.function_name}_landscape.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Landscape plot saved: {filename}")
        
        plt.show()
        return fig
    
    def create_animation(self, interval=50, save=True, fps=20):
        """
        Create animated visualization of PSO optimization
        
        Args:
            interval: int, milliseconds between frames
            save: bool, whether to save as video
            fps: int, frames per second for saved video
        
        Returns:
            animation object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # create base contour plot
        contour = ax.contour(self.X, self.Y, self.Z, levels=30, 
                            cmap='viridis', alpha=0.6)
        contourf = ax.contourf(self.X, self.Y, self.Z, levels=30, 
                              cmap='viridis', alpha=0.3)
        
        # colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Function Value', fontsize=12)
        
        # mark optimum
        if 'schwefel' in self.function_name.lower():
            optimum = [420.9687, 420.9687]
        else:
            optimum = [1.0, 1.0]
        ax.plot(optimum[0], optimum[1], 'r*', markersize=20, 
               label='Global Optimum', markeredgecolor='white', markeredgewidth=1.5)
        
        # initialize particle scatter plot
        position_history = self.pso.get_position_history()
        initial_positions = position_history[0]
        scatter = ax.scatter(initial_positions[:, 0], initial_positions[:, 1],
                           c='red', s=50, alpha=0.7, edgecolors='black', 
                           linewidth=0.5, label='Particles')
        
        # mark global best
        best_marker = ax.plot([], [], 'yo', markersize=15, 
                            label='Global Best', markeredgecolor='black', 
                            markeredgewidth=2)[0]
        
        # title with iteration counter
        title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                       ha='center', fontsize=14, fontweight='bold')
        
        # fitness text
        fitness_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', 
                             facecolor='wheat', alpha=0.8), fontsize=10)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            """Animation function called for each frame"""
            if frame < len(position_history):
                positions = position_history[frame]
                scatter.set_offsets(positions)
                
                # update global best marker
                fitness_history = self.pso.get_fitness_history()
                current_best_fitness = fitness_history[frame]
                
                # find which particle is the global best
                # (approximation - use the PSO's stored global best)
                if frame > 0:
                    best_pos = self.pso.global_best_position
                    best_marker.set_data([best_pos[0]], [best_pos[1]])
                
                # update title and fitness
                title.set_text(f'{self.function_name} - Iteration {frame}/{len(position_history)-1}')
                fitness_text.set_text(f'Best Fitness: {current_best_fitness:.6f}')
            
            return scatter, best_marker, title, fitness_text
        
        # create animation
        anim = FuncAnimation(fig, animate, frames=len(position_history),
                           interval=interval, blit=True, repeat=True)
        
        # save animation as video
        if save:
            filename = os.path.join(self.output_dir, 
                                   f'{self.function_name}_animation.mp4')
            print(f"Saving animation to {filename}...")
            print("This may take a few minutes...")
            
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(filename, writer=writer, dpi=100)
            print(f"Animation saved: {filename}")
        
        plt.tight_layout()
        return anim
    
    def create_all_visualizations(self):
        """
        Create all visualizations (convenience method)
        """
        print(f"\nCreating visualizations for {self.function_name}...")
        print("-" * 60)
        
        # convergence plot
        print("1. Creating convergence plot...")
        self.plot_convergence(save=True)
        plt.close()
        
        # landscape plot
        print("2. Creating landscape plot...")
        self.plot_landscape(save=True)
        plt.close()
        
        # animation
        print("3. Creating animation...")
        anim = self.create_animation(save=True)
        plt.close()
        
        print("-" * 60)
        print(f"All visualizations created for {self.function_name}!\n")


# test visualization
if __name__ == "__main__":
    from pso_algorithm import PSO
    from objective_functions import schwefel, rosenbrock
    
    # test with Rosenbrock (faster than schwefel)
    print("Rosenbrock visualizer test...")
    pso = PSO(
        objective_function=rosenbrock,
        bounds=(-2, 2),
        num_particles=20,
        num_iterations=50
    )
    pso.optimize(verbose=False)
    
    visualizer = PSOVisualizer(pso, "Rosenbrock_Test")
    visualizer.plot_convergence(save=False)
    visualizer.plot_landscape(save=False)