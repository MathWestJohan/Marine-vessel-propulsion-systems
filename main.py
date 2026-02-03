import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
# 1. Load data
df = pd.read_csv('Data/data.csv')
df.columns = df.columns.str.strip()

# 2. Capture the duplicates before dropping them
# keep='first' marks all but the first occurrence as a duplicate
duplicate_rows = df[df.duplicated(keep='first')]

# Save the duplicates to a list of dictionaries (if you want a Python list)
dropped_duplicates_list = duplicate_rows.to_dict('records')

# OR Save them to a CSV file to inspect them later
if not duplicate_rows.empty:
    duplicate_rows.to_csv('dropped_duplicates.csv', index=False)
    print(f"Captured {len(duplicate_rows)} duplicate rows.")
else:
    print("No duplicates found.")

# 3. Now proceed with cleaning the main dataframe
df = df.drop_duplicates()
df = df.dropna()

speed_counts = df['Ship speed (v)'].value_counts().sort_index()

print(speed_counts)

# Define the columns we want to track
gt_torque = 'Gas Turbine (GT) shaft torque (GTT) [kN m]'
prop_torques = ['Starboard Propeller Torque (Ts) [kN]', 'Port Propeller Torque (Tp) [kN]']

# Get a sorted list of all unique speeds (3, 6, 9, etc.)
unique_speeds = sorted(df['Ship speed (v)'].unique())

# Loop through each speed and generate a plot
for speed in unique_speeds:
    # Filter data for the current speed
    speed_df = df[df['Ship speed (v)'] == speed].reset_index(drop=True)

    # Create the subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Graph: GT Shaft Torque
    axes[0].plot(speed_df[gt_torque], color='red')
    axes[0].set_title(f'Gas Turbine (GT) Shaft Torque at Speed {speed}')
    axes[0].set_ylabel('Torque [kN m]')
    axes[0].grid(True)

    # Bottom Graph: Propeller Torques
    axes[1].plot(speed_df[prop_torques[0]], color='blue', label='Starboard')
    axes[1].plot(speed_df[prop_torques[1]], color='green', linestyle='--', label='Port')
    axes[1].set_title(f'Propeller Torques at Speed {speed}')
    axes[1].set_ylabel('Torque [kN]')
    axes[1].set_xlabel('Observation Index')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()


    # Close the plot to free up memory before moving to the next one
    plt.show()
speed_col = 'Ship speed (v)'
torque_col = 'Gas Turbine (GT) shaft torque (GTT) [kN m]'
fuel_col = 'Fuel flow (mf) [kg/s]'

# 2. Preparation
unique_speeds = sorted(df[speed_col].unique())
plt.close('all')  # Clear any lingering plots in memory/PyCharm cache

# 3. Automation Loop
for speed in unique_speeds:
    # Filter data for this specific speed
    speed_df = df[df[speed_col] == speed].reset_index(drop=True)

    # Create a figure with 2 subplots (top and bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Turbine Torque
    ax1.plot(speed_df[torque_col], color='tab:red', linewidth=1.5)
    ax1.set_title(f'Turbine Torque at Speed {speed}')
    ax1.set_ylabel('Torque [kN m]')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Fuel Flow Rate
    ax2.plot(speed_df[fuel_col], color='tab:orange', linewidth=1.5)
    ax2.set_title(f'Fuel Flow Rate at Speed {speed}')
    ax2.set_ylabel('Fuel flow [kg/s]')
    ax2.set_xlabel('Observation Index')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 4. Save and Clean Up
    filename = f'torque_and_fuel_speed_{speed}.png'
    plt.savefig(filename)
    print(f"Generated plot for speed {speed}: {filename}")
    plt.show()
    # Close the figure after saving to keep the PyCharm Plot tab clean
    plt.close(fig)

# Final safety clear
plt.close('all')