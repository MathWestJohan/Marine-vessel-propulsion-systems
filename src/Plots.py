import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def run_all_plots(df, image_dir):
    """
    Generate and save comprehensive visualization plots for marine vessel propulsion data.

    This function creates multiple plot types to analyze vessel propulsion parameters:
    1. A correlation heatmap showing relationships between all numerical parameters
    2. Speed-specific plots showing GT torque and propeller torques over time
    3. Speed-specific plots showing turbine torque and fuel flow rate over time

    Args:
        df (pandas.DataFrame): DataFrame containing marine vessel propulsion data with columns:
            - 'Ship speed (v)': Vessel speed values
            - 'Gas Turbine (GT) shaft torque (GTT) (kN m)': GT shaft torque measurements
            - 'Starboard Propeller Torque (Ts) (kN)': Starboard propeller torque
            - 'Port Propeller Torque (Tp) (kN)': Port propeller torque
            - 'Fuel flow (mf) (kg/s)': Fuel consumption rate
            - Other numerical parameters for correlation analysis
        image_dir (str): Directory path where generated plot images will be saved.
            Directory must exist or be creatable.

    Returns:
        None: Plots are saved to disk and displayed via plt.show()

    Side Effects:
        - Closes all existing matplotlib figures
        - Creates and saves PNG files in the specified image_dir:
            * 'correlation_heatmap.png': Correlation matrix heatmap
            * 'propeller_torque_speed_{speed}.png': Per-speed GT and propeller torque plots
            * 'torque_and_fuel_speed_{speed}.png': Per-speed turbine torque and fuel flow plots
        - Prints confirmation messages for saved files to stdout
        - Displays the correlation heatmap via plt.show()

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv('propulsion_data.csv')
        >>> run_all_plots(df, './output_images')
        Saved: ./output_images/correlation_heatmap.png
        Generated speed-based plots for 5 different speeds in ./output_images.
    """
    plt.close('all')

    # --- 1. Correlation Matrix Heatmap ---
    cols_to_drop = ['index', 'T1', 'P1']
    corr_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    corr_matrix = corr_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Marine Vessel Propulsion Plant Parameters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    heatmap_path = os.path.join(image_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved: {heatmap_path}")
    plt.show()

    # --- 2. Speed-based Torque and Fuel Plots ---
    # UPDATED: Changed [ ] to ( ) to match sanitized headers from CleaningData.py
    gt_torque_col = 'Gas Turbine (GT) shaft torque (GTT) (kN m)'
    fuel_col = 'Fuel flow (mf) (kg/s)'
    speed_col = 'Ship speed (v)'
    prop_torques = ['Starboard Propeller Torque (Ts) (kN)', 'Port Propeller Torque (Tp) (kN)']

    unique_speeds = sorted(df[speed_col].unique())

    for speed in unique_speeds:
        speed_df = df[df[speed_col] == speed].reset_index(drop=True)

        # Plot 1: GT Torque & Propeller Torques
        fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes1[0].plot(speed_df[gt_torque_col], color='red')
        axes1[0].set_title(f'Gas Turbine (GT) Shaft Torque at Speed {speed}')
        axes1[0].set_ylabel('Torque (kN m)')
        axes1[0].grid(True)

        axes1[1].plot(speed_df[prop_torques[0]], color='blue', label='Starboard')
        axes1[1].plot(speed_df[prop_torques[1]], color='green', linestyle='--', label='Port')
        axes1[1].set_title(f'Propeller Torques at Speed {speed}')
        axes1[1].set_ylabel('Torque (kN)')
        axes1[1].legend()
        axes1[1].grid(True)
        plt.tight_layout()

        fig1_path = os.path.join(image_dir, f'propeller_torque_speed_{speed}.png')
        plt.savefig(fig1_path)
        plt.close(fig1)

        # Plot 2: Turbine Torque & Fuel Flow
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(speed_df[gt_torque_col], color='tab:red', linewidth=1.5)
        ax1.set_title(f'Turbine Torque at Speed {speed}')
        ax1.set_ylabel('Torque (kN m)')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(speed_df[fuel_col], color='tab:orange', linewidth=1.5)
        ax2.set_title(f'Fuel Flow Rate at Speed {speed}')
        ax2.set_ylabel('Fuel flow (kg/s)')
        ax2.set_xlabel('Observation Index')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        fig2_path = os.path.join(image_dir, f'torque_and_fuel_speed_{speed}.png')
        plt.savefig(fig2_path)
        plt.close(fig2)

    print(f"Generated speed-based plots for {len(unique_speeds)} different speeds in {image_dir}.")