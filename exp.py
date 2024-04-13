import plotly.graph_objects as go
import os, torch
import numpy as np
import matplotlib.pyplot as plt

def viz_seg_pts(pts, pts_class):
    # -------- #
    # Plotting #
    # -------- #
    # fig = mlab.figure()
    # for c in np.unique(pts_class):
    #     ix = np.where(pts_class == c)
    #     color = tuple(list(np.random.rand(3)))
    #     mlab.points3d(pts[ix, 0], pts[ix, 1], pts[ix, 2], color=color, 
    #                   scale_factor=0.05, scale_mode='none', mode='point')
    # mlab.show()

    fig = go.Figure()

    # Generate a color map from matplotlib
    unique_labels = np.unique(pts_class)
    cmap = plt.get_cmap('tab20')  # A colormap with 20 distinct colors
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]

    # Plot each class
    for i, label in enumerate(unique_labels):
        # Select points belonging to the current class
        class_points = pts[pts_class == label]
        print(class_points.shape)
        
        # Convert matplotlib color to RGB for Plotly
        color = 'rgb' + str(tuple(int(x * 255) for x in colors[i][:-1]))
        
        # Add scatter plot for each class
        fig.add_trace(go.Scatter3d(
            x=class_points[:, 0],
            y=class_points[:, 1],
            z=class_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,  # Adjust size as needed
                color=color,  # Use class-specific color
            ),
            name=f'Class {label}'  # Label for the legend
        ))

    # Update layout for better visualization
    # fig.update_layout(
    #     title="3D Point Cloud with Class Labels",
    #     margin=dict(l=0, r=0, b=0, t=30)
    # )

    # Set equal aspect ratio for all axes to ensure proportional visualization
    # fig.update_layout(scene=dict(aspectmode='data'))

    # Show figure
    fig.show()

# for visualize
exp_name = '5obj_measure'
num_imgs = 12
saving_path = os.path.join("output/dust3r_segmented_output", f'{exp_name}_{num_imgs}.pth')
meta = torch.load(saving_path)
print(meta['dense_pt'].shape, meta['pt_class'].shape)
viz_seg_pts(meta['dense_pt'][::10], meta['pt_class'].squeeze()[::10])