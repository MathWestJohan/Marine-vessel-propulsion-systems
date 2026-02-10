import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup and Path Configuration
plt.close('all')
# Define the directory where images will be saved
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 2. Load and Initial Cleaning
df = pd.read_csv('Data/data.csv')
df.columns = df.columns.str.strip()

# 3. Capture and Handle Duplicates
# keep='first' marks all but the first occurrence as a duplicate
duplicate_rows = df[df.duplicated(keep='first')]

if not duplicate_rows.empty:
    duplicate_rows.to_csv('dropped_duplicates.csv', index=False)
    print(f"Captured {len(duplicate_rows)} duplicate rows.")
else:
    print("No duplicates found.")

# Proceed with cleaning the main dataframe
df = df.drop_duplicates()
df = df.dropna()

speed_counts = df['Ship speed (v)'].value_counts().sort_index()
print("Speed observation counts:\n", speed_counts)

# 4. Define Column Names (matching root main.py headers)
speed_col = 'Ship speed (v)'
gt_torque_col = 'Gas Turbine (GT) shaft torque (GTT) [kN m]'
prop_torques = ['Starboard Propeller Torque (Ts) [kN]', 'Port Propeller Torque (Tp) [kN]']
fuel_col = 'Fuel flow (mf) [kg/s]'

# Get sorted list of unique speeds for the loop
unique_speeds = sorted(df[speed_col].unique())

# 5. Automation Loop for Speed-Based Visualization
for speed in unique_speeds:
    # Filter data for this specific speed
    speed_df = df[df[speed_col] == speed].reset_index(drop=True)

    # --- PLOT 1: GT Shaft Torque & Propeller Torques ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Graph: GT Shaft Torque
    axes1[0].plot(speed_df[gt_torque_col], color='red')
    axes1[0].set_title(f'Gas Turbine (GT) Shaft Torque at Speed {speed}')
    axes1[0].set_ylabel('Torque [kN m]')
    axes1[0].grid(True)

    # Bottom Graph: Propeller Torques
    axes1[1].plot(speed_df[prop_torques[0]], color='blue', label='Starboard')
    axes1[1].plot(speed_df[prop_torques[1]], color='green', linestyle='--', label='Port')
    axes1[1].set_title(f'Propeller Torques at Speed {speed}')
    axes1[1].set_ylabel('Torque [kN]')
    axes1[1].set_xlabel('Observation Index')
    axes1[1].legend()
    axes1[1].grid(True)

    plt.tight_layout()
    # Save propeller torque plots to images directory
    prop_path = os.path.join(image_dir, f'propeller_torque_speed_{speed}.png')
    plt.savefig(prop_path)
    print(f"Saved: {prop_path}")
    plt.show()
    plt.close(fig1)

    # --- PLOT 2: Turbine Torque & Fuel Flow Rate ---
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Plot: Turbine Torque
    ax1.plot(speed_df[gt_torque_col], color='tab:red', linewidth=1.5)
    ax1.set_title(f'Turbine Torque at Speed {speed}')
    ax1.set_ylabel('Torque [kN m]')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Bottom Plot: Fuel Flow Rate
    ax2.plot(speed_df[fuel_col], color='tab:orange', linewidth=1.5)
    ax2.set_title(f'Fuel Flow Rate at Speed {speed}')
    ax2.set_ylabel('Fuel flow [kg/s]')
    ax2.set_xlabel('Observation Index')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Save fuel flow plots to images directory
    fuel_path = os.path.join(image_dir, f'torque_and_fuel_speed_{speed}.png')
    plt.savefig(fuel_path)
    print(f"Saved: {fuel_path}")
    plt.show()
    plt.close(fig2)

# Final safety clear
plt.close('all')