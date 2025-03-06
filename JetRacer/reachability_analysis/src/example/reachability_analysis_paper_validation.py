from pyzonotope.Zonotope import Zonotope
import numpy as np 
from pyzonotope.SafetyLayer import SafetyLayer,NLReachability
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
dim_x = 3
dim_u = 2

R0 = Zonotope(np.array([0.0,0.0,0.0]).reshape((dim_x, 1)), np.diag([0.01,0.01,0.01]))

readings = np.array([0.5, 0.9, 0.8, 1.4, 0.38, 1.0, .8 ,5.0, 0.9, 2.35]).T


Safety_layer = SafetyLayer()

plan = np.array([
    [0.1, 0.1],
    [0.2,-0.3],
    [0.1,-0.9],
    [0.6,-0.6]
])
print( len(plan))
print(plan[0])

Safety_chack_NL,obstacles,Reachable_Set_NL = Safety_layer.enforce_safety_nonlinear(R0, plan, readings)
Safety_chack,obstacles,Reachable_Set = Safety_layer.enforce_safety(R0, plan, readings)
print(Safety_chack_NL)
#print(Safety_chack)

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
    polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='lightblue', 
                             edgecolor='black', lw=2, alpha=0.5, linestyle='--')
    
    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)

# Plot the second set of Zonotopes (reachability_state_2D)
for i in range(len(Reachable_Set)):
    # Get the polygon of the Zonotope (assuming `polygon` method returns the vertices)
    X00 = Reachable_Set[i]
    center_2D = np.array(X00.center())
    generators_2D = np.array(X00.generators())
    new_center = center_2D[:2].reshape(2, 1)# First two rows of the center
    new_generators =  generators_2D[:2,:]# First two rows of the generators
    reachability_state_2D = Zonotope(new_center,new_generators)
    
     # Get the polygon vertices
    polygon_vertices = reachability_state_2D.polygon()  # Assumes this returns a 2D array of vertices
    # Create the polygon patch
    polygon_patch = Polygon(polygon_vertices.T, closed=True, facecolor='lightgreen', 
                             edgecolor='black', lw=2, alpha=0.5, linestyle='--')
    
    # Add the Polygon patch to the plot
    ax.add_patch(polygon_patch)





# Set the limits of the plot after all patches are added
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Add grid
plt.grid()

# Display all added patches in one figure

plt.show()



"""

dim_x = 2
fig, ax = plt.subplots()
center = np.array([0,0]).reshape(-1, 1)   # Center of the Zonotope
generators = [[0.25,0], [0.5, 0.001]]  # Generators of the Zonotope
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
