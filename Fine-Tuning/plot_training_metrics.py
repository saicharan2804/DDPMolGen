import os
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
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV file and save the plots as images.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing training metrics.")
    parser.add_argument('--output_dir', type=str, default='plots', help="Directory to save the output images.")
    parser.add_argument('--font_size', type=int, default=14, help="Font size for the plots.")
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

def plot_metrics(df: pd.DataFrame, metrics: list, output_dir: str, font_size: int) -> None:
    """
    Plot the specified metrics over time and save the plots as images.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
        metrics (list): List of metrics to plot.
        output_dir (str): Directory to save the output images.
        font_size (int): Font size for the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': font_size})

    for metric in metrics:
        plt.figure(figsize=(10, 8))
        sns.lineplot(data=df, x=df.index, y=metric)
        plt.title(f'{metric} over steps', fontsize=font_size + 6)
        plt.xlabel('Steps', fontsize=font_size + 6)
        plt.ylabel(metric, fontsize=font_size + 6)
        plt.xticks([])
        plt.yticks(fontsize=font_size + 6)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_over_time.png'))
        plt.close()

    print(f'All images saved in {output_dir}')

def main():
    """
    Main function to execute the script.
    """
    args = parse_arguments()
    df = load_data(args.csv_file)
    metrics = [
        'mean_reward', 'train_runtime', 'train_samples_per_second',
        'train_steps_per_second', 'train_loss', 'total_flos', 'epoch'
    ]
    plot_metrics(df, metrics, args.output_dir, args.font_size)

if __name__ == "__main__":
    main()