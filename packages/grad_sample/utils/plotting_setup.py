import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Set Seaborn aesthetic theme
sns.set_theme(style="whitegrid")

# Define a pastel color palette
# Options: 'muted', 'deep', 'bright', 'dark', 'colorblind', "Set2"
palette_choice = 'colorblind'

# pastel_palette = sns.color_palette(palette_choice)
sns.set_palette(palette_choice)

# Set matplotlib parameters
plt.rcParams.update({
    # Figure aesthetics
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 12,
    
    # Grid and background
    'axes.edgecolor': '#f0f0f0',
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8,
    'axes.facecolor': '#f9f9f9',
    
    # Axes labels and title
    'axes.labelsize': 14,
    'axes.labelcolor': '#333333',
    'axes.titlesize': 16,
    'axes.titlepad': 15,
    
    # Tick parameters
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
   
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{physics}'
})
