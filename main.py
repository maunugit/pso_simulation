import numpy as np
from pso_algorithm import PSO
from objective_functions import schwefel, rosenbrock, get_function_info
from visualization import PSOVisualizer
import config
import time
import matplotlib.pyplot as plt

def run_experiment(function_name, save_outputs = True, show_plots= True, verbose=True):
    """
    Run complete PSO experiment for a given function
    
    Args:
        function_name: str, either 'schwefel' or 'rosenbrock'
        verbose: bool, whether to print progress
    
    Returns:
        tuple: (pso object, visualizer object)
    """
    # get function info
    func_info = get_function_info(function_name)
    
    if func_info is None:
        raise ValueError(f"Unknown function: {function_name}")
    
    print("\n" + "=" * 70)
    print(f"PSO OPTIMIZATION: {func_info['name']}")
    print("=" * 70)
    
    # create PSO optimizer
    pso = PSO(
        objective_function=func_info['function'],
        bounds=func_info['bounds'],
        num_particles=config.NUM_PARTICLES,
        num_iterations=config.NUM_ITERATIONS,
        w=config.INERTIA_WEIGHT,
        c1=config.COGNITIVE_COEFF,
        c2=config.SOCIAL_COEFF
    )
    
    # run optimization
    start_time = time.time()
    best_position, best_fitness = pso.optimize(verbose=verbose)
    elapsed_time = time.time() - start_time
    
    # print results
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Function: {func_info['name']}")
    print(f"Known optimum position: {func_info['optimum']}")
    print(f"Known optimum value: {func_info['optimum_value']}")
    print(f"Found position: {best_position}")
    print(f"Found value: {best_fitness:.6f}")
    
    error = np.linalg.norm(best_position - func_info['optimum'])
    print(f"Position error: {error:.6f}")
    print(f"Optimization time: {elapsed_time:.2f} seconds")
    print("-" * 70)
    
    # create visualizations
    visualizer = PSOVisualizer(pso, func_info['name'])

    if save_outputs and not show_plots:
        # save everything in the /outputs, don't show
        visualizer.create_all_visualizations()
    elif show_plots and not save_outputs:
        # demo mode, show plots, don't save into /outputs
        visualizer.plot_convergence(save=False)
        visualizer.plot_landscape(save=False)

        # creates animation but doesn't save
        anim = visualizer.create_animation(save=False)
        plt.show() 
    elif show_plots and save_outputs:
        # full mode: both save and show
        visualizer.create_all_visualizations()
        visualizer.plot_convergence(save=False)
        visualizer.plot_landscape(save=False)
        plt.show()
    else:
        # neither save nor show (just optimize)
        print("\nSkipping visualizations completely")
    
    return pso, visualizer

def demo_mode(function_name='rosenbrock'):
    print("\n" + "=" * 70)
    print("EMO MODE - Interactive Visualization (No Saving)")
    print("=" * 70)
    print("Plots will appear in separate windows. Close them to continue.")
    print("-" * 70)
    
    if function_name.lower() == 'both':
        functions = ['rosenbrock', 'schwefel']
    else:
        functions = [function_name]
    
    for func in functions:
        pso, viz = run_experiment(func, save_outputs=False, show_plots=True, verbose=True)
        print(f"\n✓ {func.title()} demo complete!\n")
    
    print("=" * 70)
    print("Demo finished! No files were saved.")
    print("=" * 70)



def run_all_experiments(save_outputs=True):
    """
    Run PSO on both objective functions and generate all outputs
    """
    print("\n" + "=" * 70)
    print("PARTICLE SWARM OPTIMIZATION - COMPLETE EXPERIMENTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Particles: {config.NUM_PARTICLES}")
    print(f"  Iterations: {config.NUM_ITERATIONS}")
    print(f"  Inertia weight (w): {config.INERTIA_WEIGHT}")
    print(f"  Cognitive coefficient (c1): {config.COGNITIVE_COEFF}")
    print(f"  Social coefficient (c2): {config.SOCIAL_COEFF}")

    if save_outputs:
        print(f"  Mode: PRODUCTION (saving all outputs)")
    else:
        print(f"  Mode: DEMO (showing plots only)")
    
    results = {}
    
    # experiment 1: Schwefel Function
    print("\n" * 70)
    print("EXPERIMENT 1: SCHWEFEL FUNCTION")
    print(* 70)
    pso_schwefel, viz_schwefel = run_experiment('schwefel', save_outputs=save_outputs,
                                                show_plots=not save_outputs,
                                                verbose=True)
    results['schwefel'] = {
        'pso': pso_schwefel,
        'visualizer': viz_schwefel
    }
    
    # experiment 2: Rosenbrock Function
    print("\n" * 70)
    print("EXPERIMENT 2: ROSENBROCK (BANANA) FUNCTION")
    print(* 70)
    pso_rosenbrock, viz_rosenbrock = run_experiment('rosenbrock', save_outputs=save_outputs,
                                                    show_plots=not save_outputs,
                                                    verbose=True)
    results['rosenbrock'] = {
        'pso': pso_rosenbrock,
        'visualizer': viz_rosenbrock
    }
    
    # summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    if save_outputs:
        print("\nGenerated outputs in 'outputs/' directory:")
        print(" Convergence plots")
        print(" Landscape plots")
        print(" Animation videos (.mp4)")
    else:
        print("\nDemo mode - no files were saved.")
    
    print("=" * 70 + "\n")
    
    return results

def quick_test():
    """
    Quick test with fewer iterations for debugging
    """
    print("\n" + "=" * 70)
    print("QUICK TEST MODE (Fewer iterations)")
    print("=" * 70)
    
    #test on Rosenbrock with fewer iterations
    pso = PSO(
        objective_function=rosenbrock,
        bounds=(-2, 2),
        num_particles=20,
        num_iterations=50,
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    best_pos, best_fit = pso.optimize(verbose=True)
    
    # quick visualization (no video to save time)
    visualizer = PSOVisualizer(pso, "Rosenbrock_QuickTest")
    visualizer.plot_convergence(save=False)
    visualizer.plot_landscape(save=False)
    
    print("\nQuick test completed")
    return pso, visualizer


def compare_parameters():
    """
    Compare different PSO parameter settings (bonus analysis for report)
    """
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON STUDY")
    print("=" * 70)
    
    # Test different inertia weights
    inertia_values = [0.4, 0.7, 0.9]
    for w in inertia_values:
        print(f"\n--- Testing w = {w} ---")
        pso = PSO(
            objective_function=rosenbrock,
            bounds=(-2, 2),
            num_particles=30,
            num_iterations=100,
            w=w,
            c1=1.5,
            c2=1.5
        )
        best_pos, best_fit = pso.optimize(verbose=False)
        print(f"Final fitness: {best_fit:.6f}")
        print(f"Position error: {np.linalg.norm(best_pos - np.array([1.0, 1.0])):.6f}")


if __name__ == "__main__":
    import sys
    
    # checks command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'demo':
            # demo mode, just plots
            if len(sys.argv) >2:
                func = sys.argv[2].lower()
                demo_mode(func)
            else:
                demo_mode('both') # shows both by default
            quick_test()
        elif mode == 'test':
            # quick test mode
            quick_test()
        elif mode == 'compare':
            compare_parameters()
        elif mode == 'schwefel':
            run_experiment('schwefel', save_outputs=True, show_plots=False)
        elif mode == 'rosenbrock':
            run_experiment('rosenbrock', save_outputs=True, show_plots=False)
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes:")
            print("python main.py")
            print("python main.py demo")
            print("python main.py demo schwefel")
            print("python main.py demo rosenbrock")
            print("python main.py test")
            print("python main.py compare")
            print("python main.py schwefel")
            print("python mian.py rosenbrock")
    else:
        # Default: run all experiments
        print("Now running literally everything and saving")
        run_all_experiments()