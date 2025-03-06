from pyzonotope.Zonotope import Zonotope
import numpy as np 
from pyzonotope.SafetyLayer import SafetyLayer,NLReachability
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
dim_x = 2
dim_u = 2

R0 = Zonotope(np.array([-1.9,-20.0]).reshape((dim_x, 1)), np.diag([0.005,0.3]))

readings = np.array([0.5, 0.9, 0.8, 1.4, 0.38, 1.0, .8 ,5.0, 0.9, 2.35]).T


Safety_layer = SafetyLayer()

plan = np.array([
    [0.01, 0.01],
    [0.01,0.01],
    [0.01,0.01],
    [0.01,0.01]
])


Reachable_Set_NL= Safety_layer.enforce_safety_nonlinear_test(R0, plan, readings)


fig, ax = plt.subplots()
num_sets = len(Reachable_Set_NL)

# Plot the second set of Zonotopes (reachability_state_2D)
for i in range(len(Reachable_Set_NL)):
    # Get the polygon of the Zonotope (assuming `polygon` method returns the vertices)
    X00 = Reachable_Set_NL[i]
    center_2D = np.array(X00.center())
    generators_2D = np.array(X00.generators())
    new_center = center_2D[:2].reshape(2, 1)# First two rows of the center
    new_generators =  generators_2D[:2,:]# First two rows of the generators
    reachability_state_2D = Zonotope(new_center,new_generators)
    
     # Get the polygon vertices
    polygon_vertices = reachability_state_2D.polygon()  # Assumes this returns a 2D array of vertices
    # Create the polygon patch
    colors = [plt.cm.Greens(i / num_sets) for i in range(1, num_sets + 1)]  # Shades of green
    polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor=colors[i], 
                             edgecolor='black', lw=2, alpha=0.5, linestyle='--')
    
    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)


# Set the limits of the plot after all patches are added
ax.set_xlim(-2, -1.6)
ax.set_ylim(-21, -12)

# Add grid
plt.grid()

# Display all added patches in one figure

plt.show()