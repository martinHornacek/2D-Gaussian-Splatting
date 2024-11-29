import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(file_paths, column_to_plot):
    """
    Plot data from multiple CSV files with different colors and a legend.
    
    Parameters:
    file_paths (list): List of file paths to CSV files
    column_to_plot (str): Name of the column to plot
    
    Returns:
    matplotlib.figure.Figure: The created plot
    """
    # Create a new figure and set its size
    plt.figure(figsize=(10, 6))
    
    # Color palette for the plots
    colors = ['blue', 'red', 'green', 'purple']
    
    # Iterate through files and plot
    for i, file_path in enumerate(file_paths):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract the folder name from the file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # Plot the specified column
        plt.plot(df.index, df[column_to_plot], 
                 color=colors[i], 
                 label=folder_name)
    
    # Add title and labels
    plt.title('2D Gaussian Splatting')
    plt.xlabel('Epoch')
    plt.ylabel(column_to_plot)
    
    # Add legend
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    return plt

def main():
    # List of CSV file paths - replace with your actual file paths
    csv_files = [
        '/Users/martin/Documents/2D Gaussian Splatting/results/init_points_random/loss_history.csv',
        '/Users/martin/Documents/2D Gaussian Splatting/results/init_points_edges/loss_history.csv',
        '/Users/martin/Documents/2D Gaussian Splatting/results/init_points_segments/loss_history.csv',
        '/Users/martin/Documents/2D Gaussian Splatting/results/init_points_sift/loss_history.csv'
    ]
    
    # Column you want to plot - replace with your actual column name
    plot_column = 'Loss'
    
    # Create and show the plot
    plot = plot_csv_files(csv_files, plot_column)
    plot.show()

    # Optional: Save the plot
    plot.savefig('combined_plot.png')

if __name__ == '__main__':
    main()