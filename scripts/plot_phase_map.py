import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

TOLERANCE = 0.3

def plot_phase_diagram_scatter(dmi, ku):
    
    #csv_path = '..\data\saf_skyrmion_results_final.csv'
    csv_path = f'..\data\saf_skyrmion_results_final_dmi={dmi}_ku={ku}.csv'
    df = pd.read_csv(csv_path)
    
    df['skyrmion_bool'] = abs(df['S2k_bot'] - 1.0) < TOLERANCE
    filtered_df = df[ df['Ku'] == ku].reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        filtered_df[ filtered_df['skyrmion_bool'] == True]['Ms'],
        filtered_df[ filtered_df['skyrmion_bool'] == True]['D'],
        marker='o',
        color='red',
        s=200,
        label='Skyrmion'
    )

    ax.scatter(
        filtered_df[ filtered_df['skyrmion_bool'] == False]['Ms'],
        filtered_df[ filtered_df['skyrmion_bool'] == False]['D'],
        marker='x',
        color='blue',
        s=200,
        label='Other Phase'
    )

    Ms_values = filtered_df['Ms'].values
    D_values = filtered_df['D'].values
    
    ax.set_xlabel('M$_s$ [kA/m]', fontsize=12)
    ax.set_ylabel('D [nm]', fontsize=12)
    ax.set_title(f'Phase Diagram Ku={ku} DMI={dmi}', fontsize=14, weight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(sorted(set(Ms_values)))
    ax.set_yticks(sorted(set(D_values)))

    plt.tight_layout()
    plt.show()

def plot_phase_diagram_interpolate(dmi, ku):
    
    #csv_path = '..\data\saf_skyrmion_results_final.csv'
    csv_path = f'..\data\saf_skyrmion_results_final_dmi={dmi}_ku={ku}.csv'
    df = pd.read_csv(csv_path)
    
    df['skyrmion_bool'] = abs(df['S2k_bot'] - 1.0) < TOLERANCE
    filtered_df = df[ df['Ku'] == ku].reset_index()
    
    Ms = filtered_df['Ms'].values
    D = filtered_df['D'].values
    labels = filtered_df['skyrmion_bool'].astype(int).values
    
    grid_resolution = 100  # Higher resolution for smoother appearance
    Ms_grid = np.linspace(Ms.min(), Ms.max(), grid_resolution)
    D_grid = np.linspace(D.min(), D.max(), grid_resolution)
    Ms_mesh, D_mesh = np.meshgrid(Ms_grid, D_grid)

    # Interpolate the binary data onto the regular grid
    labels_grid = griddata((Ms, D), labels, (Ms_mesh, D_mesh), method='nearest')

    # Create the plot with imshow
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use imshow to display the phase diagram
    im = ax.imshow(labels_grid, 
                extent=[Ms.min(), Ms.max(), D.min(), D.max()],
                origin='lower',
                aspect='auto',
                cmap='RdBu_r',  # Red-Blue reversed colormap
                alpha=0.9,
                interpolation='bilinear')  # Options: 'nearest', 'bilinear', 'bicubic'
        
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Other Phase', 'Skyrmion'])
    cbar.set_label('Phase', rotation=270, labelpad=20, fontsize=12)

    # Labels and title
    ax.set_xlabel('Ms', fontsize=14)
    ax.set_ylabel('D', fontsize=14)
    ax.set_title(f'Phase Diagram for DMI={dmi}, Ku={ku}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='white')

    plt.tight_layout()
    plt.show()
        
def plot_phase_diagram(dmi, ku):
    
    #csv_path = '..\data\saf_skyrmion_results_final.csv'
    csv_path = f'..\data\saf_skyrmion_results_final_dmi={dmi}_ku={ku}.csv'
    df = pd.read_csv(csv_path)
    
    df['skyrmion_bool'] = abs(df['S2k_bot'] - 1.0) < TOLERANCE
    filtered_df = df[ df['Ku'] == ku].reset_index()
    
    Mss = range(260,440,20)
    Ds = range(150,825,75)
    
    data = []
    for D in Ds:
        value = []
        for Ms in Mss:
            skyrmion = filtered_df[ (filtered_df['Ms'] == Ms) & (filtered_df['D'] == D)]['skyrmion_bool'].values[0]
            value.append(int(skyrmion))
            
        data.append(value)
    
    data = np.flip(data, axis=0)
    data_matrix = np.array(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='summer', interpolation='nearest')
    
    ax.set_xticks(
        range(len(Mss)),
        labels=[f'{Ms}' for Ms in Mss]
    )
    ax.set_yticks(
        range(len(Ds)),
        labels=[f'{D}' for D in reversed(Ds)]
    )
        
    for i in range(len(Ds)):
        for j in range(len(Mss)):
            label = f'{data_matrix[i, j]:.0f}'
            text = ax.text(j, i, label, ha='center', va='center', color='black')

    ax.set_title(rf"Phase diagram, DMI={dmi} and Ku={ku}")
    ax.set_ylabel("D [nm]")
    ax.set_xlabel("Ms [kA/m]")
    
    fig.colorbar(im, ax=ax, label="Skyrmion", ticks=[0,1])
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    dmi = 0.5
    ku = 0.05
    plot_phase_diagram(dmi, ku)
    #plot_phase_diagram_interpolate(dmi, ku)