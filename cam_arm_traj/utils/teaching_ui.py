import tkinter as tk
from tkinter import filedialog,Toplevel
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from skimage.draw import polygon
from dt_apriltags import Detector
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

import torch
import copy

import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

at_detector = Detector(families='tag25h9',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

#######################
#Geometry Utils
def to_radians(degrees):
    return degrees * torch.pi / 180

def rotation_matrix(yaw, pitch, roll):
    # Convert angles to radians
    yaw, pitch, roll = to_radians(yaw), to_radians(pitch), to_radians(roll)

    # Yaw rotation matrix
    yaw_matrix = torch.tensor([
        [torch.cos(yaw), 0, torch.sin(yaw)],
        [0, 1, 0],
        [-torch.sin(yaw), 0, torch.cos(yaw)]
    ])

    # Pitch rotation matrix
    pitch_matrix = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(pitch), -torch.sin(pitch)],
        [0, torch.sin(pitch), torch.cos(pitch)]
    ])

    # Roll rotation matrix
    roll_matrix = torch.tensor([
        [torch.cos(roll), -torch.sin(roll), 0],
        [torch.sin(roll), torch.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combine the rotation matrices
    return yaw_matrix @ pitch_matrix @ roll_matrix

def rotate_vector(origin, yaw_pitch_roll):
    yaw=yaw_pitch_roll[0]
    pitch=yaw_pitch_roll[1]
    roll=yaw_pitch_roll[2]
    rot_mat = rotation_matrix(yaw, pitch, roll)
    unit_x = torch.tensor([1., 0., 0.])
    unit_z = torch.tensor([0., 0., 1.])

    # Rotate unit vectors
    rotated_x = rot_mat @ unit_x
    rotated_z = rot_mat @ unit_z

    # Normalize the vectors
    rotated_x = rotated_x / (10*torch.norm(rotated_x))
    rotated_z = rotated_z / (10*torch.norm(rotated_z))

    # Translate the vectors
    final_x = origin + rotated_x
    final_z = origin + rotated_z

    return final_x, final_z

def project_image_points_to_world_torch(img_points, mtx_s, R, T):
    world_points = []
    inv_camera_matrix = torch.linalg.inv(mtx_s)
    inv_rotation_matrix = torch.linalg.inv(R)
    for img_point in img_points:
        normalized_point = torch.matmul(inv_camera_matrix, img_point)
        point_camera = torch.tensor([normalized_point[0], normalized_point[1], 1])
        point_world = torch.matmul(inv_rotation_matrix, point_camera-T.view(-1))
        world_points.append(point_world.reshape((-1,3)))
    return torch.vstack(world_points)

def get_ray_samples(origin, targets, length, num_samples):
    directions = targets - origin
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    samples = torch.linspace(0.1, length, num_samples)  # [num_samples]
    origin = origin.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    directions = directions.unsqueeze(1)  # [n, 1, 3]
    samples = samples.unsqueeze(0).unsqueeze(2)  # [1, num_samples, 1]
    ray_samples = origin + samples * directions  # [n, num_samples, 3]
    return ray_samples

#UI utils#############
def compute_region(canvas):
    if not hasattr(canvas, 'coords'):
        return None

    # Assuming the canvas and the image are the same size
    img_shape = (canvas.winfo_height(), canvas.winfo_width())
    mask = np.zeros(img_shape, dtype=np.uint8)

    # Extract x and y coordinates
    x, y = zip(*canvas.coords)
    rr, cc = polygon(y, x)  # skimage's polygon function to get pixels within the polygon
    mask[rr, cc] = 1

    return mask

def clear_canvas(canvas):
    canvas.delete("all")
    if canvas.photo:  # Redraw the image if it exists
        canvas.create_image(0, 0, image=canvas.photo, anchor='nw')

def clear_red_lines(canvas, red_line_ids):
    # Remove all red lines from the canvas
    for line_id in red_line_ids:
        canvas.delete(line_id)
    # Clear the list after removing the lines
    red_line_ids.clear()

def clear_canvas(canvas):
    canvas.delete("all")
    if canvas.photo:  # Redraw the image if it exists
        canvas.create_image(0, 0, image=canvas.photo, anchor='nw')

def unbind_drawing(canvas):
    """ Unbind drawing events from the canvas. """
    canvas.unbind("<Button-1>")
    canvas.unbind("<ButtonRelease-1>")

def get_orientation_values(yaw_scale, pitch_scale, roll_scale):
    """ Function to retrieve values from orientation widgets. """
    #remove_arrows()
    print(f"Yaw: {yaw_scale.get()}, Pitch: {pitch_scale.get()}, Roll: {roll_scale.get()}")

def calculate_world_space_points(view_grids, camera_matrix, rotation_matrices, translation_vectors, camera_positions, ray_length, sample_points):
    world_space_points = []
    for v_grid, R, T, camera_position in zip(view_grids, rotation_matrices, translation_vectors, camera_positions):
        world_points = project_image_points_to_world_torch(v_grid[::2], camera_matrix, R, T)
        sampled_points = get_ray_samples(camera_position.reshape(-1), world_points, ray_length, sample_points)
        reshaped_points = sampled_points.reshape((-1, 3))
        world_space_points.append(reshaped_points)
    return world_space_points


def get_camera_positions(Rs, Ts):
    return [-np.dot(R.T, T) for R, T in zip(Rs, Ts)]


def find_intersection_points(w_space_points_list, ray_dist_threshold):
    # Convert all world space points to the device and reshape
    w_space_points_tor = [w_space_points.to(device).reshape((-1, 3)) for w_space_points in w_space_points_list]

    # Initialize an empty list to store intersection points
    inter_points = []

    # Loop through each combination of world space points
    for i in range(len(w_space_points_tor)):
        for j in range(i + 1, len(w_space_points_tor)):
            dists = torch.cdist(w_space_points_tor[i].half(), w_space_points_tor[j].half())
            vals_i, _ = torch.min(dists, dim=-1)
            vals_j, _ = torch.min(dists, dim=0)

            # Find the intersection points for this pair of point sets
            inter_points_i = w_space_points_tor[i][vals_i < ray_dist_threshold]
            inter_points_j = w_space_points_tor[j][vals_j < ray_dist_threshold]

            # Append the intersection points to the list
            inter_points.append(inter_points_i)
            inter_points.append(inter_points_j)

    # Concatenate all found intersection points
    return torch.cat(inter_points) if inter_points else torch.tensor([])


class TeachingUI():
    def __init__(self, root, canvas1, canvas2, status_label):
        self.canvas1 = canvas1
        self.canvas2 = canvas2
        self.status_label = status_label
        self.root = root
        self.image_path1, self.image_path2 = None, None
        self.region1, self.region2 = None, None
        self.orientation_values = {}
        self.current_cluster_id = 0
        self.Rs, self.Ts, self.Vs = None, None, None
        self.arrow_ids = {}

        self.inter_points_list=[]
        self.mean_inter_p_list=[]
        self.first_scatter_r=None
        self.second_scatter_r=None
        self.current_val_scatter=0

        self.red_line_ids_canvas1 = []
        self.red_line_ids_canvas2 = []

        # Mysterious variables
        self.dist_g=0.065
        self.width_s = 2.
        self.height_s = 3.
        self.focal_length = 3400
        self.img_size=(3008,2000)
        self.width_np=np.arange(0,self.width_s)
        self.height_np=np.arange(0,self.height_s)
        self.w, self.h = np.meshgrid(self.width_np, self.height_np, indexing='ij')
        self.grid=np.concatenate([self.w[:,:,None],self.h[:,:,None]],2).reshape((-1,2))
        self.t_grid=copy.copy(self.grid)
        self.t_grid[:,0]=((self.grid[:,0]+0.5)-0.5*self.width_s)*self.dist_g
        self.t_grid[:,1]=((self.grid[:,1]+0.5)-0.5*self.height_s)*self.dist_g
        self.t_grid_3d=np.concatenate([self.t_grid,np.zeros((len(self.t_grid),1))],axis=1)
        self.center = (self.img_size[0]/2, self.img_size[1]/2)
        self.mtx=np.array([[self.focal_length, 0, self.center[0]],
                           [0, self.focal_length, self.center[1]],
                           [0, 0, 1]], dtype = "double")
        self.distortion_coeffs = np.array([])
        self.camera_params_c=[self.focal_length,self.focal_length,1504,1000]
        self.resize_shape=(100,100)
        self.resize_transform = transforms.Resize(self.resize_shape)
        self.mtx_s=np.array([[self.focal_length/(self.img_size[0]/self.resize_shape[0]), 0, self.resize_shape[0]/2],
                    [0, self.focal_length/(self.img_size[1]/self.resize_shape[1]), self.resize_shape[1]/2],
                    [0, 0, 1]], dtype = "double")
        self.grid_2d=torch.tensor(np.indices(self.resize_shape).reshape(2, -1).T)
        self.grid_2d_w=torch.cat([self.grid_2d,torch.ones((len(self.grid_2d),1))],dim=1).double()

    def load_image(self, canvas, image_number):
        file_path = filedialog.askopenfilename()
        if file_path:
            if image_number == 1:
                self.image_path1 = file_path
            else:
                self.image_path2 = file_path
            image = Image.open(file_path)
            image = image.resize((600, 400))  # Resize the image to 1200x800
            photo = ImageTk.PhotoImage(image)
            canvas.image = image  # Keep a reference to the image
            canvas.photo = photo  # Keep a reference to the Tkinter PhotoImage
            canvas.config(width=600, height=400)
            canvas.create_image(0, 0, image=photo, anchor='nw')
            self.status_label.config(text=f"Image {image_number} loaded")

    def start_line(self, event, canvas):
        canvas.coords = [(event.x, event.y)]
        canvas.bind("<B1-Motion>", lambda e: self.on_drag(e, canvas))

    def on_drag(self, event, canvas):
        x, y = event.x, event.y
        canvas.coords.append((x, y))
        line_id = canvas.create_oval(canvas.coords[-2], canvas.coords[-1],fill='red', 
                                     outline='red', width=1)
        if canvas == self.canvas1:
            self.red_line_ids_canvas1.append(line_id)
        elif canvas == self.canvas2:
            self.red_line_ids_canvas2.append(line_id)

    def on_release(self, event, canvas):
        canvas.unbind("<B1-Motion>")
        # Close the curve if it is not closed
        if canvas.coords[0] != canvas.coords[-1]:
            canvas.coords.append(canvas.coords[0])
        line_id = canvas.create_oval(canvas.coords[-2], canvas.coords[-1], fill='red', outline='red', width=1)
        if canvas == self.canvas1:
            self.red_line_ids_canvas1.append(line_id)
        elif canvas == self.canvas2:
            self.red_line_ids_canvas2.append(line_id)

    def start_drawing(self, canvas):
        canvas.bind("<Button-1>", lambda e: self.start_line(e, canvas))
        canvas.bind("<ButtonRelease-1>", lambda e: self.on_release(e, canvas))

    def add_region(self):
        self.start_drawing(self.canvas1)
        self.start_drawing(self.canvas2)  

    def bind_drawing(self, canvas):
        """ Bind drawing events to the canvas. """
        canvas.bind("<Button-1>", lambda e: self.start_line(e, canvas))
        canvas.bind("<ButtonRelease-1>", lambda e: self.on_release(e, canvas))
    
    def draw_arrow(self, canvas, cluster_id, start_point, end_point, arrow_color='red',arrow_thickness=8):
        start_x, start_y = start_point
        end_x, end_y = end_point
        canvas_name = 'canvas1' if canvas == self.canvas1 else 'canvas2'
        # Draw an arrow from start_point to end_point
        arrow_id = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, fill=arrow_color,width=arrow_thickness)
        # Store the arrow ID
        self.arrow_ids[cluster_id][canvas_name].append((arrow_id, True))

    def toggle_arrows(self, cluster_id, visible):
        """ Toggle the visibility of arrows for a specific cluster. """
        if cluster_id in self.arrow_ids:
            for canvas_name, arrows in self.arrow_ids[cluster_id].items():
                canvas = self.canvas1 if canvas_name == 'canvas1' else self.canvas2
                for arrow_id, _ in arrows:
                    canvas.itemconfig(arrow_id, state='normal' if visible else 'hidden')
                self.arrow_ids[cluster_id][canvas_name] = [(aid, visible) for aid, _ in arrows]

    def del_arrows(self, cluster_id):
        if cluster_id in self.arrow_ids:
            for canvas_name, arrows in self.arrow_ids[cluster_id].items():
                canvas = self.canvas1 if canvas_name == 'canvas1' else self.canvas2
                for arrow_id, _ in arrows:
                    canvas.delete(arrow_id)
            #re-init dict
            self.arrow_ids[cluster_id] = {'canvas1': [], 'canvas2': []}
    

    def on_constrain_toggle(self, checkbutton,cluster_id):
        """ Handle the toggle of the check_constrain checkbox. """
        check_var = checkbutton.cget('variable')
        visible = check_var.get() == 1
        self.toggle_arrows(cluster_id, visible)

    def toggle_arrows_visibility(self, cluster_id):
        """ Toggle the visibility of arrows for the given cluster. """
        if cluster_id in self.arrow_ids:
            # Determine if any arrow in the cluster is currently visible
            currently_visible = any(visibility for _, visibility in self.arrow_ids[cluster_id]['canvas1'])

            # Toggle visibility
            new_visibility = not currently_visible
            self.toggle_arrows(cluster_id, new_visibility)

            # Update arrow visibility state
            for canvas_name in self.arrow_ids[cluster_id]:
                self.arrow_ids[cluster_id][canvas_name] = [(arrow_id, new_visibility) for arrow_id, _ in self.arrow_ids[cluster_id][canvas_name]]

    def add_region(self):
        """ Enable drawing on both canvases. """
        self.bind_drawing(self.canvas1)
        self.bind_drawing(self.canvas2)
        self.status_label.config(text="Draw regions on both images")

    def compute_both_regions(self):
        self.region1 = compute_region(self.canvas1)
        self.region2 = compute_region(self.canvas2)
        self.plot_and_clear()
        unbind_drawing(self.canvas1)
        unbind_drawing(self.canvas2)
        self.status_label.config(text="Regions computed")

    def get_rot_t(self, file_str,t_grid_3d):
        img = cv2.imread(file_str, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img, estimate_tag_pose=True,
                                camera_params=self.camera_params_c,
                                tag_size=0.05)
        tags_cent_list=[]
        for i in range(len(tags)):
            tags_cent_list.append(tags[i].center)
        tags_cent_np=np.vstack(tags_cent_list)
        success, vector_rotation, vector_translation = cv2.solvePnP(t_grid_3d,
                                                                    tags_cent_np,
                                                                    self.mtx, 
                                                                    self.distortion_coeffs)
        R,_=cv2.Rodrigues(vector_rotation)
        return (R,vector_translation,vector_rotation)

    def prepare_regions_for_processing(self, regions):
        return [self.resize_transform(torch.tensor(region).float().unsqueeze(0).unsqueeze(0)).reshape((100, 100)) for region in regions]


    def create_orientation_window(self, cluster_id):
        orientation_window = Toplevel(self.root)
        orientation_window.title("Orientation Settings")

        toggle_button = tk.Button(orientation_window, text="Toggle Arrows", command=lambda: self.toggle_arrows_visibility(cluster_id))
        toggle_button.pack()
        
        # Check if there are saved values for this cluster
        saved_values = self.orientation_values.get(cluster_id, {'yaw': 0, 'pitch': 0, 'roll': 0})

        # Yaw Scale
        yaw_scale = tk.Scale(orientation_window, from_=-180, to=180, orient='horizontal', label="Yaw")
        yaw_scale.set(saved_values['yaw'])  # Initialize with saved value
        yaw_scale.pack()

        # Pitch Scale
        pitch_scale = tk.Scale(orientation_window, from_=-90, to=90, orient='horizontal', label="Pitch")
        pitch_scale.set(saved_values['pitch'])  # Initialize with saved value
        pitch_scale.pack()

        # Roll Scale
        roll_scale = tk.Scale(orientation_window, from_=-180, to=180, orient='horizontal', label="Roll")
        roll_scale.set(saved_values['roll'])  # Initialize with saved value
        roll_scale.pack()

        # Function to update values in the dictionary
        def update_orientation_values():
            self.orientation_values[cluster_id] = {
                'yaw': yaw_scale.get(),
                'pitch': pitch_scale.get(),
                'roll': roll_scale.get()
            }
            cluster_mean=self.mean_inter_p_list[cluster_id]
            ypr=torch.tensor([yaw_scale.get(),pitch_scale.get(),roll_scale.get()])
            
            self.del_arrows(cluster_id)
            self.draw_arrows(cluster_mean, ypr, self.Vs, self.Ts, 
                             [self.canvas1, self.canvas2], cluster_id)
            #recompute arrow end in 3d
            

        # Button to get and save values
        get_values_button = tk.Button(orientation_window, text="Get and Save Values", command=update_orientation_values)
        get_values_button.pack()

    def on_scatter_point_click(self, event, cluster_id):
        print(f"Cluster {cluster_id} clicked")
        self.create_orientation_window(cluster_id)

    # Define smaller sub-functions for specific tasks

    def clear_canvases(self):
        clear_red_lines(self.canvas1, self.red_line_ids_canvas1)
        clear_red_lines(self.canvas2, self.red_line_ids_canvas2)

    def draw_arrows(self, mean_inter_p, ypr,Vs, Ts, canvases,over_ride_id=None):
        x_a, z_a = rotate_vector(mean_inter_p, ypr)
        if over_ride_id==None:
            over_ride_id=self.current_cluster_id
        all_points = torch.cat([mean_inter_p[None], x_a[None], z_a[None]], dim=0)

        for V, T, canvas in zip(Vs, Ts, canvases):
            mean, _ = cv2.projectPoints(np.array(all_points.detach().cpu()), V, T, self.mtx, 
                                        self.distortion_coeffs)
            mean = mean[:, 0] / 5
            self.draw_arrow(canvas, over_ride_id, mean[0], mean[1], 'red')
            self.draw_arrow(canvas, over_ride_id, mean[0], mean[2], 'black')

    def scatter_plot_points(self, canvas, points,size=1,colour='blue'):
        # Increment cluster ID
        cluster_id = self.current_cluster_id

        # Initialize the cluster entry
        self.arrow_ids[cluster_id] = {'canvas1': [], 'canvas2': []}

        for point in points:
            x, y = point
            scatter_id = canvas.create_oval(x-size, y-size, x+size, y+size, fill=colour, outline=colour)
            canvas.tag_bind(scatter_id, '<Button-1>', lambda e, 
                            cid=cluster_id: self.on_scatter_point_click(e, cid))
        #current_cluster_id += 1

    def project_and_draw_points(self, inter_points, Vs, Ts, canvases):
        for V, T, canvas in zip(Vs, Ts, canvases):
            scatter, _ = cv2.projectPoints(np.array(inter_points.detach().cpu()), V, T, self.mtx, 
                                           self.distortion_coeffs)
            scatter_r = (scatter / 5)[:, 0]
            self.scatter_plot_points(canvas, scatter_r)

    # Refactored main function
    def plot_and_clear(self):
        #clear_canvases([canvas1, canvas2])
        self.clear_canvases()

        self.Rs = [self.get_rot_t(self.image_path1, self.t_grid_3d)[0], 
                   self.get_rot_t(self.image_path2, self.t_grid_3d)[0]]
        self.Ts = [self.get_rot_t(self.image_path1, self.t_grid_3d)[1], 
                   self.get_rot_t(self.image_path2, self.t_grid_3d)[1]]
        self.Vs = [self.get_rot_t(self.image_path1, self.t_grid_3d)[2], 
                   self.get_rot_t(self.image_path2, self.t_grid_3d)[2]]
        camera_positions = get_camera_positions(self.Rs, self.Ts)

        mtx_s_tor = torch.tensor(self.mtx_s)
        Rs_tor = [torch.tensor(R) for R in self.Rs]
        Ts_tor = [torch.tensor(T).reshape(-1) for T in self.Ts]

        regions = [self.region1, self.region2]
        r_rs = self.prepare_regions_for_processing(regions)

        density_threshold = 0.5
        v_grids = [self.grid_2d_w[(r_r.T).reshape(-1) > density_threshold] for r_r in r_rs]

        ray_length, sample_points = 1.3, 300
        camera_positions_tor = [torch.tensor(camera_position) for camera_position in camera_positions]

        w_space_points = calculate_world_space_points(v_grids, mtx_s_tor, Rs_tor, Ts_tor, 
                                                      camera_positions_tor, ray_length, 
                                                      sample_points)

        ray_dist_threshold = 0.005
        inter_points = find_intersection_points(w_space_points, ray_dist_threshold)
        if(len(inter_points)==0):
            print("no intersetion!")
            return
        self.project_and_draw_points(inter_points, self.Vs, self.Ts, [self.canvas1, self.canvas2])

        mean_inter_p = inter_points.mean(axis=0).cpu().detach()
        
        self.mean_inter_p_list.append(mean_inter_p.cpu().detach().clone())
        
        #initialise to 0 ypr
        init_ypr=torch.tensor([0.,0.,0.])
        self.draw_arrows(mean_inter_p, init_ypr, self.Vs, self.Ts, [self.canvas1, self.canvas2])
        #set arrows to invisible first
        self.toggle_arrows_visibility(self.current_cluster_id)
        self.inter_points_list.append(inter_points.cpu().detach().clone())
        self.current_cluster_id += 1

##############
### Calculations
def create_tensor_from_dict(key, data_dict):
    if key in data_dict:
        values = data_dict[key]
        return torch.tensor([values['yaw'], values['pitch'], values['roll']])
    else:
        return None
def dist_set(x, set_g):
    pos = x[:3]  # Position part
    goal_pos = set_g[0]
    pos_dists = torch.norm(pos - torch.mean(goal_pos,dim=0), dim=-1)

    if set_g[1] is not None:
        ori = x[3:]  # Orientation part
        goal_ori = set_g[1]
        ori_dists = torch.norm(ori - goal_ori, dim=-1)
    else:
        ori_dists = 0  # No orientation goal

    dist_set_mean = pos_dists + ori_dists
    min_dist = pos_dists
    min_ori_dist = ori_dists

    return dist_set_mean, min_dist, min_ori_dist

def velocity(coord, set_g):
    if not isinstance(coord, torch.Tensor):
        coord = torch.tensor(coord, dtype=torch.float32)
    coord.requires_grad = True
    output, _, _ = dist_set(coord, set_g)  # Compute the 'dist_set' function
    output.backward()  # Perform backpropagation to compute the gradient
    return coord.grad  # Return the gradient

def euler_integrator(initial_pos, initial_ori, goal_pos, goal_ori=None, time_step=0.005, steps=2000, stop_threshold=0.01):
    pos = torch.tensor(initial_pos, dtype=torch.float32)
    coord = torch.cat((pos, torch.tensor(initial_ori, dtype=torch.float32))) if goal_ori is not None else pos
    positions = []  # List to store positions and orientations
    for _ in range(steps):
        coord.requires_grad = True
        grad = velocity(coord, (goal_pos, goal_ori)).detach()
        _, current_pos_dist, current_ori_dist = dist_set(coord, (goal_pos, goal_ori))
        if goal_ori is not None:
            total_dist = current_pos_dist + current_ori_dist
            pos_scale = (current_pos_dist / total_dist).clamp(min=0.1)
            ori_scale = (current_ori_dist / total_dist).clamp(min=0.1)

            pos_grad = grad[:3] * pos_scale
            ori_grad = grad[3:] * ori_scale

            # Update position and orientation
            new_pos = coord[:3] - time_step * pos_grad
            new_ori = coord[3:] - time_step * ori_grad
            coord = torch.cat((new_pos, new_ori), dim=0)
        else:
            # Update position only
            pos_grad = grad * current_pos_dist
            new_pos = coord - time_step * pos_grad
            coord = new_pos
        coord = torch.tensor(coord, requires_grad=True)
        positions.append(coord.tolist())
        _, min_dist, min_ori_dist = dist_set(coord, (goal_pos, goal_ori))
        if min_dist < stop_threshold and (min_ori_dist < stop_threshold or goal_ori is None):
            break
        positions_tor=torch.tensor(positions)
        #add if No orientation
        if(goal_ori==None):
            positions_tor=torch.cat([positions_tor,
                                     torch.ones_like(positions_tor)*initial_ori.reshape((1,3))],dim=1)
    return positions_tor

def get_to_goal_ind(init_pos,init_ori,inter_points_l,ind,ori_values):
    goal_pos=inter_points_l[ind]
    goal_ori=create_tensor_from_dict(ind, ori_values)
    if(goal_ori!=None):
        goal_ori=to_radians(goal_ori)
    trajectory = euler_integrator(init_pos, init_ori, goal_pos, goal_ori)
    return(trajectory)

def multi_goal_inter_list(init_pos,init_ori,inter_list,ori_values):
    cur_pos=init_pos.clone()
    cur_ori=init_ori.clone()
    all_traj_list=[]
    for i in range(len(inter_list)):
        trajectory=get_to_goal_ind(cur_pos, cur_ori,inter_list,i,ori_values)
        cur_pos=trajectory[-1,:3]
        cur_ori=trajectory[-1,3:]
        all_traj_list.append(trajectory)
    return(torch.cat(all_traj_list))
