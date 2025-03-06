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

dim_x = 3
dim_u = 2



initpoints =500
steps = 1
totalsamples = initpoints*steps
X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.3 * np.diag(np.ones((dim_x, 1)).T[0]))
U = Zonotope(np.array(np.ones((dim_u, 1))*10),0.25 * np.diag(np.ones((dim_u, 1)).T[0]))
W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.005 * np.ones((dim_x, 1)))


GW = []
for i in range(W.generators().shape[1]):
    vec = np.reshape(W.Z[:, i + 1], (dim_x, 1))
    dummy = []
    dummy.append(np.hstack((vec, np.zeros((dim_x, totalsamples - 1)))))
    for j in range(1, totalsamples, 1):
        right = np.reshape(dummy[i][:, 0:j], (dim_x, -1))
        left = dummy[i][:, j:]
        dummy.append(np.hstack((left, right)))
    GW.append(np.array(dummy))

GW = np.array(GW)

Wmatzono = MatZonotope(np.zeros((dim_x, totalsamples)), GW)



# Load the saved files
U_full = np.load('U_full_NL.npy')
X_0T = np.load('X_0T_NL.npy')
X_1T = np.load('X_1T_NL.npy')



x0 = X0.center()


index_0 =1
index_1 =1
AB = get_AB(U_full, X_0T, X_1T, Wmatzono)

total_steps = 4
center_2D = X0.center()
generators_2D = X0.generators()
new_center = center_2D[:2, :]  # First two rows of the center
new_generators = generators_2D[:2, :]  # First two rows of the generators
reachability_state_2D = [Zonotope(new_center,new_generators)]
reachability_state = [X0]
for i in range(total_steps):
    reachability_state[i] = reachability_state[i].reduce('girard', 100)
    new_state = AB * reachability_state[i].cart_prod(U) + W
    reachability_state.append(new_state.reduce('girard', 1))
    
    reachability_state_2D[i] = reachability_state_2D[i].reduce('girard', 100)
    center_2D = new_state.center()
    generators_2D = new_state.generators()
    new_center = center_2D[:2, :]  # First two rows of the center
    new_generators = generators_2D[:2, :]  # First two rows of the generators
    new_state_2D = Zonotope(new_center,new_generators)
    reachability_state_2D.append(new_state_2D)





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


Safety_chack,obstacles = Safety_layer.enforce_safety_nonlinear(Zonotope(new_center,new_generators), plan_holder, readings)

print(Safety_chack)



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