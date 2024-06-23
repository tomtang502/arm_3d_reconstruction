import numpy as np
import pyvista as pv
import torch, os

exp_name = '4obj_measure'
num_imgs = 12
saving_path = os.path.join("output/dust3r_segmented_output", f'{exp_name}_{num_imgs}.pth')
meta = torch.load(saving_path)
# Example data generation
points = meta['dense_pt'].numpy()  # Random points in a 100x100x100 cube
colors = meta['colors']  # Random colors

def pick_callback(picker):   
    print(f"Coordinates: {picker}")


# Create a PyVista Point Cloud
point_cloud = pv.PolyData(points)
point_cloud['colors'] = colors  # Add colors to the point cloud

# Plotting
plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=2)
plotter.enable_point_picking(pick_callback, show_message=True)
plotter.show_grid(color='black')
plotter.show()
