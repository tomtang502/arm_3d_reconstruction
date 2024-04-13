import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# Example data
n = 100  # Number of points
points = np.random.rand(n, 3)  # Random points in 3D space
labels = np.random.randint(0, 4, size=(n,))  # Random labels from 0 to 3

# Initialize figure
fig = go.Figure()

# Colors for each class (you can customize these)
colors = ['blue', 'green', 'red', 'yellow']

# Plot each class
for i, color in enumerate(colors):
    # Select points belonging to the current class
    class_points = points[labels == i]
    
    # Add scatter plot for each class
    fig.add_trace(go.Scatter3d(
        x=class_points[:, 0],
        y=class_points[:, 1],
        z=class_points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color,  # Use class-specific color
        ),
        name=f'Class {i}'  # Label for the legend
    ))

# Update layout for better visualization
fig.update_layout(
    title="3D Point Cloud with Class Labels",
    margin=dict(l=0, r=0, b=0, t=30)
)

# Set equal aspect ratio for all axes
fig.update_layout(scene=dict(aspectmode='data'))

# Show figure
fig.show()
