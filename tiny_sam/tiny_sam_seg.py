import numpy as np
import torch, threading, fnmatch
import matplotlib.pyplot as plt
import cv2, os
from data_label import ImageLabel, mask_viz

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    

def show_one_image(img, mask, min_idx, max_idx, map_name_dict, pixel_map, pixel_viz, hint=''):
    tmp_img = np.ones((mask.shape[0], mask.shape[1], 3))
    tmp_img = np.copy(img)
    color_mask = np.array([0, 245, 245])
    tmp_img[mask] = color_mask
    
    
    def get_user_input():
        while True:
            try:
                user_input = int(input(hint+"Please enter the class idx between {} and {}: ".format(min_idx, max_idx)))
                if min_idx <= user_input <= max_idx:
                    print(f"Classified as {map_name_dict[user_input][1]}")
                    break  # Exits the loop
                else:
                    print("Error: The number is not within the range. Try again.")
            except ValueError:
                print("Error: That's not an integer idx. Please enter a valid integer idx.")
        return user_input
    
    plt.imshow(tmp_img)
    plt.show(block=False)
    cid = get_user_input()
    plt.close()
    
    pixel_map[mask] = cid
    pixel_viz[mask] = map_name_dict[cid][0]
    return pixel_map, pixel_viz
        


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from tinysam import sam_model_registry, SamHierarchicalMaskGenerator

    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint="./weights/tinysam.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    sam.eval()
    # Specify the parent directory path
    images_dir = "images"
    sub_dirs = [os.path.join(images_dir, name) for name in os.listdir(images_dir)
                if os.path.isdir(os.path.join(images_dir, name))]
    print(sub_dirs)

    # Specify the experiment
    ############## Experiment Name Here ################
    exp_name = "shelf"
    ############## Experiment Name Here ################
    img_labels = ImageLabel()
    img_map = img_labels.class_names[exp_name]
    idxs = img_map.keys()
    idx_hint = ''
    for idx in img_map: idx_hint += f'{str(idx)}: {img_map[idx][1]}\n'
    min_idx, max_idx = min(idxs), max(idxs)
    target_sub_dirs = [name for name in sub_dirs if exp_name in name]
    for dir in target_sub_dirs:
        jpg_files = []
        for file in os.listdir(dir):
            if fnmatch.fnmatch(file, '*.jpg'):
                jpg_files.append(os.path.join(dir, file))
        i = 0
        n = len(jpg_files)
        for img_path in jpg_files:
            mask_generator = SamHierarchicalMaskGenerator(sam)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image.shape = (480, 640, 3)
            # num_mask = 31 (480, 640)
   
            masks = mask_generator.hierarchical_generate(image)
        
            print(len(masks), masks[0]['segmentation'].shape)
            good = False
            while not good:
                print(f"image{i + 1} / {n}")
                pixel_map = np.zeros((image.shape[0], image.shape[1]))
                pixel_viz = np.ones((image.shape[0], image.shape[1], 4))
                for mask in masks:
                    pixel_map, pixel_viz = show_one_image(image, mask['segmentation'], min_idx, max_idx, img_map, 
                                                        pixel_map, pixel_viz, hint=idx_hint)    
                pixel_map = pixel_map.astype(int)
                plt.figure(figsize=(10,10))
                plt.imshow(pixel_viz)
                plt.show()
                plt.imshow(mask_viz(pixel_map))
                plt.show()
                while True:
                    try:
                        is_good = int(input(f"Does the classification looks good for image {img_path}\n1: good, next\n0: shit, redo\n"))
                        if 0 <= is_good <= 1:
                            break  # Exits the loop
                        else:
                            print("Error: The number is not within the range. Try again.")
                    except ValueError:
                        print("Error: That's not an integer idx. Please enter a valid integer idx.")
                good = bool(is_good)
            
            out_path = img_path[:-len('.jpg')] + '.pth'
            tensors = {
                'pix_cla': torch.tensor(pixel_map),
                'viz': torch.tensor(pixel_viz)
            }
            torch.save(tensors, out_path)
            i += 1
        print(f"Done with directory {dir}, drag it to output/tiny_sam_output to save progress!")

