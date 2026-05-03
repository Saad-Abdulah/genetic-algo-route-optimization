# Route Optimization using Genetic Algorithm

AI2002 - Artificial Intelligence  
Assignment 3 Implementation Repository  
FAST NUCES Islamabad, Spring 2026

## 1) Project Summary

This project solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA).  
Each candidate route is a permutation of city indices. The solver minimizes total closed-tour distance using Haversine distance between latitude and longitude coordinates.

Core GA operators:
- Tournament selection
- Order crossover (OX1)
- Swap mutation
- Elitism

## 2) Repository Structure

```text
route_optimizer_app/
  main.py
  benchmark.py
  genetic_algorithm.py
  route_utils.py
  visualization.py
  requirements.txt
  README.md
  data/
    cities.csv
    cities_20.csv
    cities_25.csv
  results/                  # created automatically on run
```

## 3) Environment and Dependencies

- Python 3.10 or higher
- matplotlib

Install:

```bash
cd route_optimizer_app
source /Users/saad/Desktop/Python/Virtual/venv/bin/activate
pip install -r requirements.txt
```

## 4) How to Run

### 4.1 Interactive Solver

```bash
cd route_optimizer_app
source /Users/saad/Desktop/Python/Virtual/venv/bin/activate
python main.py
```

Inputs requested at runtime:
- CSV path (default: `data/cities.csv`)
- Population size
- Mutation rate
- Generations
- Tournament size
- Optional random seed

Outputs:
- Console summary with best route and total distance
- `results/best_route.png`
- `results/fitness_plot.png`

### 4.2 GUI App (for screenshots)

```bash
cd route_optimizer_app
source /Users/saad/Desktop/Python/Virtual/venv/bin/activate
streamlit run gui_app.py
```

GUI features:
- Built-in dataset selection (`cities_20.csv`, `cities_25.csv`, `cities.csv`)
- CSV upload support
- GA parameter sliders
- Error handling for inconsistent data
- Route table + route plot + convergence plot
- Download buttons for plot artifacts

### 4.3 Benchmark Runner (for execution evidence)

```bash
cd route_optimizer_app
source /Users/saad/Desktop/Python/Virtual/venv/bin/activate
python benchmark.py
```

This runs fixed configurations on:
- `data/cities_20.csv`
- `data/cities_25.csv`
- `data/cities.csv` (expanded set)

It prints dataset size, best distance, and runtime for each case.

## 5) Input Format

CSV columns (case-insensitive):
- `City` or `city`
- `Latitude` or `latitude`
- `Longitude` or `longitude`

Example:

```csv
City,Latitude,Longitude
New York,40.7128,-74.0060
Chicago,41.8781,-87.6298
Los Angeles,34.0522,-118.2437
```

## 6) Implementation Notes

- Distances are precomputed in a matrix once per run.
- Fitness evaluation is done once per generation and reused during parent selection.
- Route validity is checked before final reporting.
- Visualization uses non-interactive backend (`Agg`) for stable execution.

## 7) Reproducibility

For reproducible results:
- Use the same dataset
- Set fixed GA parameters
- Provide the same random seed in `main.py` prompt

## 8) Authors

- Amir Shahzad (23i-2011)
- Saad Abdullah (23i-3045)
