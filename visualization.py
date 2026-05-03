import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_route(route, cities, save_path=None):
    """
    Plot the optimized route on a 2D scatter map.
    Cities are shown as red dots, route as blue lines,
    and the starting city is highlighted with a green star.
    """
    ordered_lats = [cities[i].lat for i in route]
    ordered_lons = [cities[i].lon for i in route]

    # close the loop back to starting city
    ordered_lats.append(ordered_lats[0])
    ordered_lons.append(ordered_lons[0])

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(ordered_lons, ordered_lats, "b-o",
            linewidth=1.4, markersize=6, markerfacecolor="red",
            markeredgecolor="darkred", alpha=0.8)

    # draw arrows to indicate direction
    for i in range(len(ordered_lats) - 1):
        dx = ordered_lons[i + 1] - ordered_lons[i]
        dy = ordered_lats[i + 1] - ordered_lats[i]
        ax.annotate("", xy=(ordered_lons[i + 1], ordered_lats[i + 1]),
                     xytext=(ordered_lons[i], ordered_lats[i]),
                     arrowprops=dict(arrowstyle="->", color="steelblue",
                                     lw=1.2))

    # label each city
    for city in cities:
        ax.annotate(city.name, (city.lon, city.lat),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=7, color="black")

    # highlight start/end city
    start = cities[route[0]]
    ax.scatter([start.lon], [start.lat], c="limegreen", s=200,
               zorder=10, marker="*", edgecolors="darkgreen",
               label="Start / End")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Optimized Route - Genetic Algorithm (TSP)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Route plot saved -> {save_path}")

    plt.close(fig)


def plot_fitness(history, save_path=None):
    """
    Plot how the best distance evolves across generations.
    Shows convergence behaviour of the genetic algorithm.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(history)), history, color="royalblue",
            linewidth=1.5, label="Best Distance")

    # simple moving average to show the trend more clearly
    window = max(1, len(history) // 20)
    if window > 1 and len(history) > window:
        moving_avg = []
        for i in range(len(history)):
            start_idx = max(0, i - window + 1)
            avg = sum(history[start_idx:i + 1]) / (i - start_idx + 1)
            moving_avg.append(avg)
        ax.plot(range(len(moving_avg)), moving_avg, color="tomato",
                linewidth=1.2, linestyle="--", label=f"Moving Avg (w={window})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Fitness Convergence Over Generations")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Fitness plot saved -> {save_path}")

    plt.close(fig)
