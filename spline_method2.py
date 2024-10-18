import numpy as np
import matplotlib.pyplot as plt

# Example waypoints
waypoints = np.array([
    [0, 0, 0],
    [1, 2, 0],
    [3, 5, 1],
    [6, 10, 2],
    [8, 5, 5],
    [10, 2, 4],
    [12, 0, 5]
])


def calculate_b_spline_control_points(waypoints):
    n = len(waypoints) - 1  # Number of waypoints minus 1
    P = waypoints

    # Initialize the B-spline control points
    B = np.zeros_like(P)
    B[0] = P[0]
    B[-1] = P[-1]

    # Matrix setup for solving the B-spline control points
    A = np.zeros((n+1, n+1))
    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i-1] = 1 / 6
        A[i, i] = 2 / 3
        A[i, i+1] = 1 / 6

    # Solving the system
    B = np.linalg.solve(A, P)
    return B

B = calculate_b_spline_control_points(waypoints)

# Generate the B-spline curve
def bspline_curve(control_points, num_points=100):
    def de_boor(k, x, t, c, p):
        """
        de Boor's algorithm to compute B-spline curves
        k: index of knot interval that contains x
        x: point
        t: knot vector
        c: control points
        p: degree of B-spline
        :return: curve point at x
        """
        d = [c[j + k - p] for j in range(0, p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        return d[p]

    n = len(control_points) - 1
    p = 3  # degree of B-spline
    t = np.array([0, 0, 0, 0] + list(range(1, n - p + 2)) + [n - p + 1, n - p + 1, n - p + 1, n - p + 1], dtype=float)
    t /= t[-1]  # normalize knot vector

    curve = np.zeros((num_points, control_points.shape[1]))
    for i in range(num_points):
        x = i / (num_points - 1)
        k = np.searchsorted(t, x) - 1
        point = de_boor(k, x, t, control_points, p)
        curve[i] = point

    return curve

curve = bspline_curve(B)

# Plot the waypoints and B-spline curve
def plot_trajectory(waypoints, curve):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot waypoints
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'ro-', label='Waypoints')

    # Plot B-spline curve
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', label='B-spline Curve')

    ax.legend()
    plt.show()

plot_trajectory(waypoints, curve)
