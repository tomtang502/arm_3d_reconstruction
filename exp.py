from PIL import Image

def crop_center(image, target_width=512, target_height=384):
    width, height = image.size
    
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    
    cropped_image = image.crop((left, top, left+target_width, top+target_height))
    return cropped_image

# Usage
cropped_img = crop_center(Image.open('arm_captured_images/3obj_measure/3obj_measure_j.jpg').convert('RGB'))  # Replace with the path to your image file
cropped_img.show()
