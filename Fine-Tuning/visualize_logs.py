import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV file.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing training metrics.")
    parser.add_argument('--output_file', type=str, default='Epoch_15_fex_V3.png', help="Path to save the output plot image.")
    parser.add_argument('--metrics', type=str, nargs='+', default=[
        'mean_reward', 'train_runtime', 'train_samples_per_second',
        'train_steps_per_second', 'train_loss', 'total_flos', 'epoch'
    ], help="List of metrics to plot.")
    return parser.parse_args()

def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load the CSV file into a DataFrame and preprocess it.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def plot_metrics(df: pd.DataFrame, metrics: list, output_file: str) -> None:
    """
    Plot the specified metrics over time and save the plot.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        metrics (list): List of metrics to plot.
        output_file (str): Path to save the output plot image.
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.lineplot(data=df, x=df.index, y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} over Time')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(metric)
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to execute the script.
    """
    args = parse_arguments()
    df = load_data(args.csv_file)
    plot_metrics(df, args.metrics, args.output_file)

if __name__ == "__main__":
    main()