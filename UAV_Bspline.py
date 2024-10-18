import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import time

# Define waypoints 
waypoints = np.array([
    [0.0, 0.0],   # Start point
    [1.0, 2.0],   # Waypoint 1
    [2.0, 0.0],   # Waypoint 2
    [3.0, 3.5],   # Waypoint 3
    [4.0, 2.0]    # End point
])

# Define many small obstacles as circles (using center and radius)
obstacles = [
    {'center': [0.5, 1.0], 'radius': 0.2},
    {'center': [1.5, 1.5], 'radius': 0.2},
    {'center': [2.0, 2.5], 'radius': 0.1},
    {'center': [2.5, 1.0], 'radius': 0.3},
    {'center': [3.0, 2.0], 'radius': 0.2},
    {'center': [3.5, 1.5], 'radius': 0.1}
]

# Function to adjust waypoints to avoid obstacles
def adjust_waypoints(waypoints, obstacles, clearance=0.4):      #increased clearance here
    new_waypoints = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        new_waypoints.append(p1)
        
        for obs in obstacles:
            center = np.array(obs['center'])
            radius = obs['radius'] + clearance
            
            # Check if the segment intersects the obstacle
            d = np.linalg.norm(np.cross(p2 - p1, p1 - center)) / np.linalg.norm(p2 - p1)
            if d < radius:
                # Add a new waypoint to detour around the obstacle
                direction = np.cross(p2 - p1, np.array([0, 0, 1]))[:2] 
                #cross product is taken to avoid obstacle and move in a direction perpendicular to XY plane
                direction /= np.linalg.norm(direction)
                new_point = center + direction * radius
                new_waypoints.append(new_point)
                
    new_waypoints.append(waypoints[-1])
    return np.array(new_waypoints)


# Degree of the B-spline
degree = 3
# Initialize UAV position (start from the first waypoint)
uav_position = waypoints[0]

for i in range(len(waypoints) - 1):
    # Adjust waypoints dynamically
    current_waypoints = waypoints[i:]
    adjusted_waypoints = adjust_waypoints(current_waypoints, obstacles)

    # Number of adjusted waypoints
    n_adjusted_waypoints = len(adjusted_waypoints)
    
    if n_adjusted_waypoints < degree + 1:
        continue  # Skip if there are not enough waypoints for the spline

    # Define knots
    knots = np.concatenate((
        np.zeros(degree), 
        np.arange(n_adjusted_waypoints - degree + 1),
        np.full(degree, n_adjusted_waypoints - degree)
    ))


    # Create the B-spline
    spline = BSpline(knots, adjusted_waypoints, degree)

    # Generate points on the B-spline
    t = np.linspace(0, n_adjusted_waypoints - degree, 100)
    spline_points = spline(t)

    # Plot the B-spline and adjusted waypoints
    plt.plot(spline_points[:, 0], spline_points[:, 1], label=f'UAV Path Segment {i+1}')
    # plt.plot(adjusted_waypoints[:, 0], adjusted_waypoints[:, 1], 'o-', label=f'Adjusted Waypoints {i+1}')


#PLOTIING ALL DATA 

# Plot obstacles
for obs in obstacles:
    circle = plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.5)
    plt.gca().add_artist(circle)

#inital and final positions
plt.scatter(uav_position[0], uav_position[1], color='red', zorder=5, label='UAV Position')  # Initial UAV position
plt.scatter(waypoints[-1][0], waypoints[-1][1], color='green', zorder=5, label='UAV final Position')  # final UAV position

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('UAV Path Planning with Dynamic B-spline and Obstacles')
plt.axis('equal')
plt.show()