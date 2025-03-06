from pyzonotope.Zonotope import Zonotope
from pyzonotope.MatZonotope import MatZonotope
import numpy as np 
from pyzonotope.read_matlab import read_matlab
from pyzonotope.reachability_analysis import  get_AB
import joblib
import scipy.signal as scipysig
from pyzonotope.SafetyLayer import SafetyLayer
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
dim_x = 5
initpoints =120
steps = 1
totalsamples = initpoints*steps
R0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.1 * np.diag(np.ones((dim_x, 1)).T[0]))


readings = np.array([0.5, 0.9, 0.8, 1.4, 0.38, 1.0, .8 ,5.0, 0.9, 2.35]).T
plan = np.array([0.1, 0.2, 0.5, 1.0])
Safety_layer = SafetyLayer()


plan = np.array([
    [.1, .2],
    [.4, .5],
    [.7, .8],
    [.10, .11]
])

plan_holder = np.zeros((10, 8)).astype(np.float32)
plan_holder[:4, :2]  = plan


Safety_chack,obstacles,Reachable_Set = Safety_layer.enforce_safety(R0, plan_holder, readings)

#print(Safety_chack)


"""
dim_x = 2
fig, ax = plt.subplots()
center = np.array([0,0]).reshape(-1, 1)   # Center of the Zonotope
generators = [[0.0,0], [0, 0.001]]  # Generators of the Zonotope
X00 = Zonotope(center, generators)  # Create the Zonotope object

    # Get the polygon of the Zonotope (assuming `polygon` method returns the vertices)
polygon_vertices = X00.polygon()

    # Create the polygon object using matplotlib's Polygon class
    # Assuming polygon_vertices is a 2D array where each row is a vertex
polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='lightblue', edgecolor='black', lw=2, alpha=0.5, linestyle='-')

    # Add the Polygon patch to the plot
ax.add_patch(polygon_patch)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

# Add grid
plt.grid()

# Display all added patches in one figure
plt.show()

"""

"""


# Create the plot
fig, ax = plt.subplots()

# Plot the first set of Zonotopes (obstacles)
for i in range(len(obstacles)):
    # Define the Zonotope object X0
    dim_x = 2  # assuming a 2D Zonotope, modify this as needed
    center = obstacles[i].center()  # Center of the Zonotope
    generators = obstacles[i].generators()  # Generators of the Zonotope

    X0 = Zonotope(center, generators)  # Create the Zonotope object

    # Get the polygon of the Zonotope (assuming `polygon` method returns the vertices)
    polygon_vertices = X0.polygon()

    # Create the polygon object using matplotlib's Polygon class
    # Assuming polygon_vertices is a 2D array where each row is a vertex
    polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='lightblue', edgecolor='black', lw=2, alpha=0.5, linestyle='-')

    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)

# Plot the second set of Zonotopes (reachability_state_2D)
for i in range(len(reachability_state_2D)):
    # Get the polygon of the Zonotope (assuming `polygon` method returns the vertices)
    polygon_vertices = reachability_state_2D[i].polygon()

    # Create the polygon object using matplotlib's Polygon class
    # Assuming polygon_vertices is a 2D array where each row is a vertex
    polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='lightgreen', edgecolor='black', lw=2, alpha=0.5, linestyle='--')

    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)

# Set the limits of the plot after all patches are added
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

# Add grid
plt.grid()

# Display all added patches in one figure
plt.show()
"""