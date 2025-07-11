import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_data(json_file):
    """Load benchmark data from a JSON file"""
    with open(json_file, "r") as f:
        return json.load(f)


def extract_times(data):
    """Extract GPU and Python overhead times from benchmark data"""
    times = {}
    # Get algorithm name from the first benchmark
    algorithm = data["benchmarks"][0]["name"].split("/")[0]
    for bench in data["benchmarks"]:
        # Extract size from the benchmark name
        size = int(bench["name"].split("/")[-1])
        # Convert nanoseconds to microseconds
        gpu_time = bench["gpu_time"] * 1e-3  # ns to μs
        python_overhead = bench["python_overhead"] * 1e-3  # ns to μs
        total_time = bench["total_time"] * 1e-3  # ns to μs
        times[size] = {
            "gpu_time": gpu_time,
            "python_overhead": python_overhead,
            "total_time": total_time,
        }
    return algorithm, times


def create_plot(
    algorithm_name,
    baseline_times,
    low_overhead_times,
    output_file="overhead_comparison.png",
):
    sizes = [2**20, 2**26]
    x = np.arange(len(sizes)) * 1.5  # Increase spacing between size groups
    width = 0.3  # Reduce bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each size, create stacked bars
    for i, size in enumerate(sizes):
        # Baseline bars
        baseline_total = baseline_times[size]["total_time"]
        baseline_gpu = baseline_times[size]["gpu_time"]
        baseline_overhead = baseline_times[size]["python_overhead"]

        # Convert to percentages
        baseline_gpu_pct = (baseline_gpu / baseline_total) * 100
        baseline_overhead_pct = (baseline_overhead / baseline_total) * 100

        # Low overhead bars
        low_total = low_overhead_times[size]["total_time"]
        low_gpu = low_overhead_times[size]["gpu_time"]
        low_overhead = low_overhead_times[size]["python_overhead"]

        # Convert to percentages
        low_gpu_pct = (low_gpu / low_total) * 100
        low_overhead_pct = (low_overhead / low_total) * 100

        # Create stacked bars with same colors
        # Baseline
        ax.bar(
            x[i] - width,
            baseline_gpu_pct,
            width,
            label="GPU Time" if i == 0 else "",
            color="#2ecc71",
        )
        ax.bar(
            x[i] - width,
            baseline_overhead_pct,
            width,
            bottom=baseline_gpu_pct,
            label="Python Overhead" if i == 0 else "",
            color="#e74c3c",
        )

        # Low overhead
        ax.bar(x[i] + width, low_gpu_pct, width, color="#2ecc71")
        ax.bar(
            x[i] + width, low_overhead_pct, width, bottom=low_gpu_pct, color="#e74c3c"
        )

        # Add percentage labels on the bars
        ax.text(
            x[i] - width,
            baseline_gpu_pct / 2,
            f"{baseline_gpu_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            x[i] - width,
            baseline_gpu_pct + baseline_overhead_pct / 2,
            f"{baseline_overhead_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
        )

        ax.text(
            x[i] + width,
            low_gpu_pct / 2,
            f"{low_gpu_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            x[i] + width,
            low_gpu_pct + low_overhead_pct / 2,
            f"{low_overhead_pct:.1f}%",
            ha="center",
            va="center",
            color="white",
        )

        # Add "Baseline" and "Low Overhead" labels above the bars with total time
        max_height_baseline = baseline_gpu_pct + baseline_overhead_pct
        max_height_low = low_gpu_pct + low_overhead_pct

        # Calculate speedup
        speedup = baseline_total / low_total

        ax.text(
            x[i] - width,
            max_height_baseline + 2,
            "Baseline\n{:.1f} μs\n".format(baseline_total),
            ha="center",
            va="bottom",
        )
        ax.text(
            x[i] + width,
            max_height_low + 2,
            "Low Overhead\n{:.1f} μs\n{:.2f}x speedup".format(low_total, speedup),
            ha="center",
            va="bottom",
        )

    # Format algorithm name for title
    algorithm_title = algorithm_name.replace("_", " ").title()

    # Customize the plot
    ax.set_ylabel("Percentage of Total Time (%)")
    ax.set_title(f"{algorithm_title}: GPU Time vs Python Overhead Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"2^{int(np.log2(size))}" for size in sizes])
    ax.set_xlabel("Number of elements")
    ax.legend()
    ax.set_ylim(0, 120)  # Increased more to make room for three lines of text

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate overhead comparison plot from benchmark JSON files"
    )
    parser.add_argument(
        "baseline_json", help="JSON file containing baseline benchmark results"
    )
    parser.add_argument(
        "low_overhead_json", help="JSON file containing low overhead benchmark results"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="overhead_comparison.png",
        help="Output file name for the plot (default: overhead_comparison.png)",
    )

    args = parser.parse_args()

    # Load benchmark data from JSON files
    baseline_data = load_benchmark_data(args.baseline_json)
    low_overhead_data = load_benchmark_data(args.low_overhead_json)

    # Extract timing information and algorithm name
    algorithm_name, baseline_times = extract_times(baseline_data)
    _, low_overhead_times = extract_times(low_overhead_data)

    # Create and save the plot
    create_plot(algorithm_name, baseline_times, low_overhead_times, args.output)


if __name__ == "__main__":
    main()
