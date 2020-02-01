import os
import cv2
import util
import numpy as np

window_src_dir = 'F:/ece5831/windows/Ricky/groupings/original'
output_dir = 'F:/ece5831/windows/Ricky/groupings/separated'

def read_image(path):
    image = cv2.imread(path)
    return image.astype(np.float32)

def find_images():
    images = []

    for root, dirs, files in os.walk(window_src_dir):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))         
    
    masks = []
    for img in images:
        mask_filepath = img[0:len(img) - 4] + '.png'
        mask_filepath = mask_filepath[::-1]
        mask_filepath = mask_filepath.replace('_tas_', '_ksam_', 1)
        mask_filepath = mask_filepath[::-1]
        masks.append(mask_filepath)
    
    return (images, masks)

def separate_image(image_path, mask_path):
    out_file_path = get_save_location(image_path)

    if os.path.exists(out_file_path):
        print("File already exists %s" % out_file_path)
        return
    
    try:
        f = open(out_file_path, 'w')
        f.close()
    except:
        return
    
    print(image_path, mask_path)
    
    image = read_image(image_path)
    mask = read_image(mask_path)

    ## make array same shape filled with zeros
    separated = np.zeros_like(mask) 

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            pixels = mask[x][y]
            if pixels[0] != 0.0 or pixels[1] != 0.0 or pixels[2] != 0.0:
                separated[x][y] = image[x][y]    
                
    cv2.imwrite(out_file_path, separated)
    print("separated ::: ", out_file_path)

def get_save_location(image_path):
    save_dir = create_output_folder(image_path)
    file_name = os.path.basename(image_path)
    file_path = os.path.join(save_dir, file_name + ".bmp")

    return file_path

def create_output_folder(image_path):
    src_directory = os.path.dirname(image_path)
    (head, tail) = os.path.split(src_directory)
        
    output_sub_dir = os.path.join(output_dir, tail)
    
    if os.path.isdir(output_sub_dir) is False:
        os.mkdir(output_sub_dir)

    return output_sub_dir

##
## run the code
##
#util.create_dir_or_is_empty(output_dir)

(images, masks) = find_images()

for i in range(0, len(images)):
    separate_image(images[i], masks[i])
