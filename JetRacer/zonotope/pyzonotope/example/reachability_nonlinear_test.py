from pyzonotope.Zonotope import Zonotope
import numpy as np 
from pyzonotope.SafetyLayer import SafetyLayer, NLReachability
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

# Define dimensions
dim_x = 4
dim_u = 2
yaw =   -np.pi/2 * 0
x = 0.0
y = 0.0
current_position = np.array([x,y]).reshape(2,1)
# Initial zonotope
R0 = Zonotope(np.array([x, y, np.cos(yaw), np.sin(yaw)]).reshape((dim_x, 1)), np.diag([0.25, 0.25, 0.01, 0.01]))

# Sensor readings
readings = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1]).T
zeros_array1 = np.ones(1) * 3
zeros_array2 = np.ones(27) * 3
new_readings = np.concatenate(( readings, zeros_array2))

# Motion plan
plan = np.array([
    [1.5, -1.5],
    [1.5, -1.5],
    [1.5, -1.5],
    [1.5, -1.5],
    [1.5, -1.5],
    [1.5, -1.5],





])

Safety_layer = SafetyLayer()

start_time_Reachability = time.time()


# Enforce safety
Reachable_Set_NL, obstacles, _, _ = Safety_layer.enforce_safety_nonlinear_test(R0, plan, new_readings, -yaw,current_position)
print(len(obstacles))

stop_time_Reachability = time.time()
Duration_Reachability = stop_time_Reachability - start_time_Reachability

print(f"Duration_Reachability Average: {Duration_Reachability:.4f} seconds")

        #print(safety_check)

# Create figure
fig, ax = plt.subplots()

# Function to apply the coordinate transformation
def rotate_coordinates(coords):
    return np.array([-coords[1], coords[0]])

# Plot obstacles
for i, obstacle in enumerate(obstacles):
    center = obstacle.center()
    generators = obstacle.generators()

    # Rotate center and generators
    rotated_center = rotate_coordinates(center)
    rotated_generators = np.array([rotate_coordinates(gen) for gen in generators.T]).T

    # Create rotated zonotope
    X0 = Zonotope(rotated_center.reshape(2, 1), rotated_generators)

    # Get polygon of rotated zonotope
    polygon_vertices = X0.polygon()

    # Create polygon patch
    polygon_patch = Polygon(
        polygon_vertices.T, closed=True, facecolor='none',
        edgecolor='red', lw=1 + i * 0.5, alpha=1.0, linestyle='-'
    )
    ax.add_patch(polygon_patch)

# Plot reachable sets
num_sets = len(Reachable_Set_NL)
for i, X00 in enumerate(Reachable_Set_NL):
    center_2D = np.array(X00.center())
    generators_2D = np.array(X00.generators())

    # Rotate coordinates
    rotated_center = rotate_coordinates(center_2D[:2])
    rotated_generators = np.array([rotate_coordinates(gen) for gen in generators_2D.T]).T

    # Create rotated zonotope
    reachability_state_2D = Zonotope(rotated_center.reshape(2, 1), rotated_generators)

    # Get polygon vertices
    polygon_vertices = reachability_state_2D.polygon()

    # Color logic
    color = 'red' if i == 0 else plt.cm.Greens(i / num_sets)
    line = '-' if i == 0 else '--'

    # Create polygon patch
    polygon_patch = Polygon(
        polygon_vertices.T, closed=True, facecolor=color,
        edgecolor='black', lw=2, alpha=0.5, linestyle=line
    )
    ax.add_patch(polygon_patch)

# Adjust axis to match new orientation
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Labels for new coordinate system
ax.set_xlabel("+y (left)")
ax.set_ylabel("+x (up)")

# Aspect ratio and grid
ax.set_aspect('equal')
plt.grid()
plt.show()
