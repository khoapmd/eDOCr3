import os
from eDOCr import tools
import cv2
import string
from skimage import io
import numpy as np
import tensorflow as tf
import warnings

# Suppress TensorFlow deprecation warnings and info messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Optimize TensorFlow for better performance
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

dest_DIR='tests/test_Results'
file_path='tests/test_samples/Candle_holder.jpg'
filename=os.path.splitext(os.path.basename(file_path))[0]

# Load and resize image with good quality
img = cv2.imread(file_path)
print(f"Original image size: {img.shape}")

# Resize to a reasonable size that maintains readability and model compatibility
h, w = img.shape[:2]
max_size = 1200

# Ensure dimensions are divisible by 32 (common requirement for deep learning models)
def make_divisible_by_32(size):
    return ((size + 31) // 32) * 32

if max(h, w) > max_size:
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
else:
    new_h, new_w = h, w

# Ensure dimensions are divisible by 32 for model compatibility
new_h = make_divisible_by_32(new_h)
new_w = make_divisible_by_32(new_w)

# Resize the image
img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
print(f"Resized image to: {img.shape} (model-compatible dimensions)")

# Validate image dimensions
if img is None or img.size == 0:
    raise ValueError("Failed to load or process image")
if img.shape[0] < 32 or img.shape[1] < 32:
    raise ValueError(f"Image too small for processing: {img.shape}. Minimum size is 32x32 pixels.")

# Use the CORRECT alphabets that match the pretrained models
# The pretrained model expects: string.digits + string.ascii_lowercase
DEFAULT_ALPHABET = string.digits + string.ascii_lowercase

# For engineering drawings, we need to map our custom characters to the default alphabet
# This is a simplified approach - in practice, you'd want to train custom models
alphabet_dimensions = DEFAULT_ALPHABET  # Use default alphabet for now
alphabet_infoblock = DEFAULT_ALPHABET   # Use default alphabet for now  
alphabet_gdts = DEFAULT_ALPHABET        # Use default alphabet for now

# Use the pretrained models without custom weights
model_dimensions = None  # Use pretrained model
model_infoblock = None   # Use pretrained model
model_gdts = None        # Use pretrained model

color_palette={'infoblock':(180,220,250),'gdts':(94,204,243),'dimensions':(93,206,175),'frame':(167,234,82),'flag':(241,65,36)}
cluster_t=20

print("Step 1: Finding rectangles...")
class_list, img_boxes=tools.box_tree.findrect(img)
print(f"Found {len(class_list)} rectangles")

print("Step 2: Processing rectangles...")
boxes_infoblock,gdt_boxes,cl_frame,process_img=tools.img_process.process_rect(class_list,img)
io.imsave(os.path.join(dest_DIR, filename+'_process_correct.jpg'),process_img)
print("Processed image saved")

print("Step 3: Reading info blocks...")
infoblock_dict=tools.pipeline_infoblock.read_infoblocks(boxes_infoblock,img,alphabet_infoblock,model_infoblock)
print(f"Extracted {len(infoblock_dict)} info blocks")

print("Step 4: Reading GD&T...")
gdt_dict=tools.pipeline_gdts.read_gdtbox1(gdt_boxes,alphabet_gdts,model_gdts,alphabet_dimensions,model_dimensions )
print(f"Extracted {len(gdt_dict)} GD&T elements")

process_img=os.path.join(dest_DIR, filename+'_process_correct.jpg')

print("Step 5: Reading dimensions...")
dimension_dict=tools.pipeline_dimensions.read_dimensions(process_img,alphabet_dimensions,model_dimensions,cluster_t)
print(f"Extracted {len(dimension_dict)} dimensions")

print("Step 6: Creating mask...")
mask_img=tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict,cl_frame,color_palette)

#Record the results
print("Step 7: Saving results...")
io.imsave(os.path.join(dest_DIR, filename+'_boxes_correct.jpg'),img_boxes)
io.imsave(os.path.join(dest_DIR, filename+ '_mask_correct.jpg'),mask_img)

tools.output.record_data(dest_DIR,filename+'_correct',infoblock_dict, gdt_dict,dimension_dict)

print("Test with correct alphabet completed successfully!")
print(f"Results saved with '_correct' suffix")
print("This version uses the pretrained model's expected alphabet for better accuracy")
