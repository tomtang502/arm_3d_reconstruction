from utils.teaching_ui import *
from utils.traj_visual import *
from move_arm import *

sample_name = "23triangle0"
use_sample = True
initial_pos = torch.tensor([0.42, 0, 0.2]) #xyx in arm angle 0.42, 0, 0.2 in meter
# img init loc fron will [0.2, -0.2, 0.15]
initial_ori = torch.tensor([0.0, 0.0, 0.0])

if use_sample:
    all_traj, inter_points_list, orientation_values = torch.load(f'sample_motion/{sample_name}.pt')
else:
    root = tk.Tk()
    root.title("Diagrammatic Teaching as Inverse Motion Planning")

    # Create a frame for buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, fill=tk.X)

    # Status label
    status_label = tk.Label(root, text="Select an action", relief=tk.SUNKEN, anchor='w')
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Create canvases for displaying images
    canvas1 = tk.Canvas(root, width=600, height=400)
    canvas1.pack(side="left", padx=10, pady=10)

    canvas2 = tk.Canvas(root, width=600, height=400)
    canvas2.pack(side="right", padx=10, pady=10)

    tu = TeachingUI(root, canvas1, canvas2, status_label)

    # Buttons for loading images
    button1 = tk.Button(button_frame, text="Load Image 1", 
                        command=lambda: tu.load_image(canvas1, 1))
    button1.pack(side="left", padx=10, pady=10)

    button2 = tk.Button(button_frame, text="Load Image 2", 
                        command=lambda: tu.load_image(canvas2, 2))
    button2.pack(side="left", padx=10, pady=10)

    # Button for adding regions
    add_region_button = tk.Button(button_frame, text="Add Region", command=tu.add_region)
    add_region_button.pack(side="left", padx=10, pady=10)

    # Button for computing regions
    compute_region_button = tk.Button(button_frame, text="Compute Region", 
                                    command=tu.compute_both_regions)
    compute_region_button.pack(side="left", padx=10, pady=10)

    root.mainloop()

    
    # goal_pos=inter_points_list[0]
    # goal_ori=to_radians(torch.tensor([90.0, -90.0, 0.0]))
    # goal_ori=None

    # trajectory=get_to_goal_ind(initial_pos, initial_ori,inter_points_list,1,orientation_values)
    all_traj=multi_goal_inter_list(initial_pos, initial_ori, 
                                tu.inter_points_list, tu.orientation_values)
    inter_points_list, orientation_values = tu.inter_points_list, tu.orientation_values
    print(f"--------Caculated Trajectory with shape of ({all_traj.shape})-------")

    pts = [all_traj, tu.inter_points_list, tu.orientation_values]
    name = "triangle0ori"
    torch.save(pts, f'sample_motion/23{name}.pt')


for i in range(0, all_traj.shape[0]):
    print(all_traj[i])
    #print(orientation_values[i])
print(all_traj.shape) #orientation_values.shape)
all_traj = convert_tag_to_arm_coordinates(all_traj, [0.21, 0.0, 0.1])
# buggy, see notes
# show_3d_traj(inter_points_list, all_traj[:,3:6])
# print(orientation_values)
show_3d_kpt(inter_points_list)