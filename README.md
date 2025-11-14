# PSO simulation
PSO implementation for optimizing Schwefel and Rosenbrock banana functions with matplotlib animated visualizations

### 1. Clone the repo
git clone
cd pso_simulation

### 2. Install dependencies
pip install -r requirements.txt

For video generation, you need FFmpeg:
On windows (powershell admin): `choco install ffmpeg` or from [ffmpeg.org](https://ffmpeg.org/download.html)

### 3. Running the simulation
Full run:
python main.py

Demo mode:
python main.py demo

Quick test:
python main.py test

"Available modes:"
            "python main.py"
            "python main.py demo"
            "python main.py demo schwefel"
            "python main.py demo rosenbrock"
            "python main.py test"
            "python main.py compare"
            "python main.py schwefel"
            "python mian.py rosenbrock"

### 4. Edit config.py to adjust PSO parameters

### 5. After running, check /outputs for animations and plot visualizations