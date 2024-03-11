import numpy as np
from scipy.spatial.transform import Rotation

def poses_convert_to_viz(poses):
    OPENGL = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    transform = np.linalg.inv(poses[0] @ OPENGL @ rot)
    converted_poses = []
    for pose_c2w in poses:
        rot45 = np.eye(4)
        rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
        # rot45[2, 3] = -height  # set the tip of the cone = optical center
        # aspect_ratio = np.eye(4)
        # aspect_ratio[0, 0] = W/H 
        #  @ aspect_ratio 
        #pose_c2w = pose_c2w @ OPENGL@ rot45
        new_pose_c2w = np.dot(np.linalg.pinv(transform), pose_c2w)
        converted_poses.append(new_pose_c2w)
    return np.stack(converted_poses, axis=0)


# cams2world = scene.get_im_poses().cpu()
# pts3d = to_numpy(scene.get_pts3d())

# pts3d = to_numpy(pts3d)
# imgs = to_numpy(imgs)
# focals = to_numpy(focals)
# cams2world = to_numpy(cams2world)

# for i, pose_c2w in enumerate(cams2world):
#     add_scene_cam(scene, pose_c2w, camera_edge_color,
#                     None if transparent_cams else imgs[i], focals[i],
#                     imsize=imgs[i].shape[1::-1], screen_width=cam_size)
# # rot = np.eye(4)
# # rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
# # scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
# # outfile = os.path.join(outdir, 'scene.glb')
# # print('(exporting 3D scene to', outfile, ')')
# # scene.export(file_obj=outfile)
# # return outfile


# def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03):

#     if image is not None:
#         H, W, THREE = image.shape
#         assert THREE == 3
#         if image.dtype != np.uint8:
#             image = np.uint8(255*image)
#     elif imsize is not None:
#         W, H = imsize
#     elif focal is not None:
#         H = W = focal / 1.1
#     else:
#         H = W = 1

#     if focal is None:
#         focal = min(H, W) * 1.1  # default value
#     elif isinstance(focal, np.ndarray):
#         focal = focal[0]

#     # create fake camera
#     height = focal * screen_width / H
#     width = screen_width * 0.5**0.5
#     rot45 = np.eye(4)
#     rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
#     rot45[2, 3] = -height  # set the tip of the cone = optical center
#     aspect_ratio = np.eye(4)
#     aspect_ratio[0, 0] = W/H
#     transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
#     cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

#     # this is the image
#     if image is not None:
#         vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
#         faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
#         img = trimesh.Trimesh(vertices=vertices, faces=faces)
#         uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
#         img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
#         scene.add_geometry(img)

#     # this is the camera mesh
#     rot2 = np.eye(4)
#     rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
#     vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
#     vertices = geotrf(transform, vertices)
#     faces = []
#     for face in cam.faces:
#         if 0 in face:
#             continue
#         a, b, c = face
#         a2, b2, c2 = face + len(cam.vertices)
#         a3, b3, c3 = face + 2*len(cam.vertices)

#         # add 3 pseudo-edges
#         faces.append((a, b, b2))
#         faces.append((a, a2, c))
#         faces.append((c2, b, c))

#         faces.append((a, b, b3))
#         faces.append((a, a3, c))
#         faces.append((c3, b, c))

#     # no culling
#     faces += [(c, b, a) for a, b, c in faces]

#     cam = trimesh.Trimesh(vertices=vertices, faces=faces)
#     cam.visual.face_colors[:, :3] = edge_color
#     scene.add_geometry(cam)