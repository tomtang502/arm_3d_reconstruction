import numpy as np
import pyvista as pv

# Generate some synthetic point cloud data
num_points = 1000
points = np.random.rand(num_points, 3) * 100  # Random points within a 100x100x100 cube
point_cloud = pv.PolyData(points)
point_cloud['scalars'] = np.random.rand(num_points)  # Just some random scalars for coloring

def pick_callback(picker):
    point_id = picker#.point_id
    print(point_id.shape)
    if True: #point_id != -1:  # Check if a valid point was picked
        picked_point = point_id#points[point_id]
        print(f"Picked point ID: {point_id}, Coordinates: {picked_point}")

# Create a plotter and add the point cloud
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars='scalars', color=True, point_size=5)

# Enable point picking
plotter.enable_point_picking(pick_callback, show_message=True)
plotter.show_grid(color='black')

# Display the plotter window
plotter.show()
