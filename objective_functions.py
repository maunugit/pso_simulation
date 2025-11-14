import numpy as np


def schwefel(position):
    """
    Schwefel function - complex landscape with many local minima
    def schwefel(x, y):
        return 418.9829*2 - x*sin(√|x|) - y*sin(√|y|)
    
    Global minimum: f(x*) = 0, at x* = (420.9687, 420.9687)
    Search domain: typically [-500, 500] for each dimension

    Takes a 2D NumPy array [x, y] and returns a scalar.
    """
    x, y = position[0], position[1]
    dim = 2
    return 418.9829 * dim - (x * np.sin(np.sqrt(np.abs(x))) + 
                              y * np.sin(np.sqrt(np.abs(y))))


def rosenbrock(position):
    """
    Rosenbrock function (Banana function) - curved valley shape
    def rosenbrock(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2
    
    Global minimum: f(1, 1) = 0
    Because:
    rosenbrock(1, 1) = (1 - 1)² + 100*(1 - 1²)²
                 = (0)² + 100*(0)²
                 = 0 + 0
                 = 0
    This creates a valley-like shape.    
    Search domain: typically [-2, 2] for each dimension
    """
    x, y = position[0], position[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def get_function_info(function_name):
    """
    Returns a small metadata dict for either "schwefel" or "rosenbrock".
    Metadata has:
        - the function itself
        - search bounds
        - true global optimum pos and value
        - a nice human-readable name :)
    """
    info = {
        'schwefel': {
            'function': schwefel,
            'bounds': (-500, 500),
            'optimum': np.array([420.9687, 420.9687]),
            'optimum_value': 0.0,
            'name': 'Schwefel Function'
        },
        'rosenbrock': {
            'function': rosenbrock,
            'bounds': (-2, 2),
            'optimum': np.array([1.0, 1.0]),
            'optimum_value': 0.0,
            'name': 'Rosenbrock (Banana) Function'
        }
    }
    return info.get(function_name.lower())


# Test functions
if __name__ == "__main__":
    # Test Schwefel
    schwefel_opt = np.array([420.9687, 420.9687])
    print(f"Schwefel at optimum: {schwefel(schwefel_opt):.6f}")
    
    # Test Rosenbrock
    rosenbrock_opt = np.array([1.0, 1.0])
    print(f"Rosenbrock at optimum: {rosenbrock(rosenbrock_opt):.6f}")
    
    # Test random points
    random_point = np.array([0.0, 0.0])
    print(f"\nSchwefel at (0,0): {schwefel(random_point):.6f}")
    print(f"Rosenbrock at (0,0): {rosenbrock(random_point):.6f}")