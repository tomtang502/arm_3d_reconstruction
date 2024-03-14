import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

def get_xyz_lim(xrange, yrange, zrange):
    x_min, x_max = xrange
    y_min, y_max = yrange
    z_min, z_max = zrange

    # Calculate the ranges for X, Y, and Z axes
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    z_diff = z_max - z_min

    # Find the maximum range among X, Y, and Z axes
    max_range = max(x_diff, y_diff, z_diff)

    # Calculate the center for each axis
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Calculate the limits for each axis
    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    y_min = y_center - max_range / 2
    y_max = y_center + max_range / 2
    z_min = z_center - max_range / 2
    z_max = z_center + max_range / 2
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


# pred_vert is a seq of mesh, and meshidx specify which mesh to graph
# passing in 3d keypoints to visualize 3d key points
def graph_single_struct(xyz_stack, line_connect=False, s=50):
    x = xyz_stack[:, 0]
    y = xyz_stack[:, 1]
    z = xyz_stack[:, 2]
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1,1,1])
    # Scatter plot
    ax.scatter(x, y, z, c='r', marker='o', s=s)

    if line_connect:
    # Connect points with lines
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], c='b')

    # Label the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set the limits for each axis
    
    xrange_lim, yrange_lim, zrange_lim = get_xyz_lim((np.min(x), np.max(x)), 
                                                     (np.min(y), np.max(y)), 
                                                     (np.min(z), np.max(z)))
    ax.set_xlim(xrange_lim)
    ax.set_ylim(yrange_lim)
    ax.set_zlim(zrange_lim)
    # Show the plot
    plt.show()


def graph_double_struct(xyz1_stack, xyz2_stack, line_connect=False):
    x = xyz1_stack[:, 0]
    y = xyz1_stack[:, 1]
    z = xyz1_stack[:, 2]
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1,1,1])
    # Scatter plot
    ax.scatter(x, y, z, c='r', marker='o', s=150)

    x = xyz2_stack[:, 0]
    y = xyz2_stack[:, 1]
    z = xyz2_stack[:, 2]
    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o', s=150)


    if line_connect:
    # Connect points with lines
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], c='b')

    # Label the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set the limits for each axis
    # xrange_lim, yrange_lim, zrange_lim = get_xyz_lim((np.min(x), np.max(x)), 
    #                                                  (np.min(y), np.max(y)), 
    #                                                  (np.min(z), np.max(z)))
    # ax.set_xlim(xrange_lim)
    # ax.set_ylim(yrange_lim)
    # ax.set_zlim(zrange_lim)
    # Show the plot
    plt.show()


color_names = [
    'red',
    'blue',
    'green',
    'yellow',
    'cyan',
    'magenta',
    'black',
    'white',
    'gray',
    'brown',
    'orange',
    'pink',
    'purple',
    'lavender',
    'maroon',
    'navy',
    'teal',
    'turquoise',
    'gold',
    'silver'
]

"""
[In] struct_list: a list of structure to graph (all with N*3 shape)
[In] struct_names: a list of structure names corresponds to the previous input
[In] dot_size_list: a list of float specify relative dot size between structures
"""
def plotty_graph_multistruct(struct_list, struct_names, dot_size_list):
    fig = go.Figure()
    color_num = len(color_names)
    for i in range(len(struct_list)):
        var = struct_list[i] 
        fig.add_trace(go.Scatter3d(x=var[:, 0],  # X coordinates
            y=var[:, 1],  # Y coordinates
            z=var[:, 2],  # Z coordinates
            mode='markers', marker=dict(size=dot_size_list[i], color=color_names[i%len(color_names)]),
            name=struct_names[i]))

    title = ''
    for struct_name in struct_names:
        title += struct_name + " + "
    title = title[:-3]
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title=title)

    fig.update_layout(width=1000, height=700)
    fig.show()