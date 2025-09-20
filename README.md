# eDOCr

A specialized OCR (Optical Character Recognition) system for digitizing mechanical engineering drawings, powered by [keras-ocr](https://github.com/faustomorales/keras-ocr).

[![Python Version](https://img.shields.io/badge/python-≥3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-≥2.0.0-orange.svg)](https://www.tensorflow.org/)
[![Version](https://img.shields.io/badge/version-0.0.4-green.svg)](https://pypi.org/project/eDOCr/)

## Features

- **Specialized Engineering Recognition**: Optimized for mechanical engineering drawings
- **Multi-Component Detection**:
  - Dimensions and measurements
  - GD&T (Geometric Dimensioning and Tolerancing) symbols
  - Information blocks and title boxes
  - Technical annotations
- **Advanced Processing**:
  - Automatic text orientation detection
  - Smart clustering of related elements
  - Watermark removal capability
  - High-quality image preprocessing
- **Customizable Models**: Train on your specific engineering fonts and symbols
- **Multiple Output Formats**: Structured data export in CSV format

## Installation

### Prerequisites
- Tested with Python 3.13.1
- [uv](https://github.com/astral-sh/uv) (recommended for faster package management)

### Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/khoapmd/eDOCr
cd eDOCr
```

#### 2. Create and Activate a Virtual Environment
```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

#### 3. Install eDOCr
```bash
uv pip install -r requirements.txt
uv pip install .
```

## Usage

### Command Line Interface

Process drawings directly from the command line:

```bash
python eDOCr/ocr_it.py path/to/drawing.pdf
```

Options:
```bash
--dest-folder PATH  # Output directory (default: current directory)
--water            # Enable watermark removal
--cluster N        # Threshold distance for grouping (default: 20px)
```

### Quick Start with test_drawing.py

For a quick test of the system's capabilities, you can use the provided `test_drawing.py`:

```bash
# Run using uv (recommended)
uv run python tests/test_drawing.py

# Or using regular Python if uv is not installed
python tests/test_drawing.py
```

### Basic Python API

```python
import os
import string
from eDOCr import tools
import cv2
from skimage import io

# 1. Setup paths
dest_dir = 'output'
file_path = 'input/drawing.jpg'
filename = os.path.splitext(os.path.basename(file_path))[0]

# 2. Load image
img = cv2.imread(file_path)

# 3. Configure recognition
# Define symbols for different annotation types
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'  # GD&T symbols
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'         # Feature Control Frame symbols
Extra = '(),.+-±:/°"⌀'           # Additional symbols

# Setup recognition alphabets
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
alphabet_infoblock = string.digits + string.ascii_letters + ',.:-/'
alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols

# 4. Process the drawing
class_list, img_boxes = tools.box_tree.findrect(img)
boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)

# 5. Extract information
dimension_dict = tools.pipeline_dimensions.read_dimensions(
    process_img, 
    alphabet_dimensions, 
    model_dimensions='models/recognizer_dimensions.h5'
)

# 6. Save results
io.imsave(os.path.join(dest_dir, f'{filename}_boxes.jpg'), img_boxes)
tools.output.record_data(dest_dir, filename, dimension_dict)
```

### Optimized Usage Example

Here's a complete example with performance optimizations and detailed progress logging:

```python
import os
from eDOCr import tools
import cv2
import string
from skimage import io
import numpy as np
import tensorflow as tf
import warnings

# Optimize TensorFlow performance
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Setup paths
dest_DIR = 'tests/test_Results'
file_path = 'tests/test_samples/LIU0010.jpg'
filename = os.path.splitext(os.path.basename(file_path))[0]

# Load and optimize image
img = cv2.imread(file_path)
print(f"Original image size: {img.shape}")

# Auto-resize for optimal processing
h, w = img.shape[:2]
max_size = 1200
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    print(f"Resized image to: {img.shape}")

# Configure recognition with default pretrained settings
DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
alphabet_dimensions = DEFAULT_ALPHABET
alphabet_infoblock = DEFAULT_ALPHABET
alphabet_gdts = DEFAULT_ALPHABET

# Process with detailed logging
print("Step 1: Finding rectangles...")
class_list, img_boxes = tools.box_tree.findrect(img)
print(f"Found {len(class_list)} rectangles")

print("Step 2: Processing rectangles...")
boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
io.imsave(os.path.join(dest_DIR, filename + '_process.jpg'), process_img)

print("Step 3: Reading info blocks...")
infoblock_dict = tools.pipeline_infoblock.read_infoblocks(boxes_infoblock, img, alphabet_infoblock)
print(f"Extracted {len(infoblock_dict)} info blocks")

print("Step 4: Reading GD&T...")
gdt_dict = tools.pipeline_gdts.read_gdtbox1(gdt_boxes, alphabet_gdts)
print(f"Extracted {len(gdt_dict)} GD&T elements")

print("Step 5: Reading dimensions...")
dimension_dict = tools.pipeline_dimensions.read_dimensions(process_img, alphabet_dimensions)
print(f"Extracted {len(dimension_dict)} dimensions")

print("Step 6: Saving results...")
tools.output.record_data(dest_DIR, filename, infoblock_dict, gdt_dict, dimension_dict)
print("Processing completed successfully!")
```

The test script will:
1. Process a sample drawing (`tests/test_samples/LIU0010.jpg`)
2. Apply performance optimizations automatically
3. Generate output files in `tests/test_Results/`:
   - `*_boxes.jpg`: Shows detected rectangles
   - `*_process.jpg`: Shows preprocessing results
   - `*_mask.jpg`: Final visualization with annotations
   - CSV files containing extracted data

This is an excellent way to verify your installation and see the system in action with optimized settings.

## Custom Model Training

> Note: Font files are only included when installing from source.

```python
import os
import string
from eDOCr import keras_ocr
from eDOCr.keras_ocr_models import train_recognizer

# 1. Configure training
alphabet = string.digits + 'AaBCDRGHhMmnx' + '().,+-±:/°"⌀'
samples = 10000
recognizer_basepath = 'eDOCr/keras_ocr_models/models'

# 2. Prepare training data
backgrounds = ['eDOCr/keras_ocr_models/backgrounds/0.jpg'] * samples
fonts = [
    f'eDOCr/keras_ocr_models/fonts/{f}'
    for f in os.listdir('eDOCr/keras_ocr_models/fonts')
]

# 3. Train model (optionally use pretrained model)
train_recognizer.generate_n_train(
    alphabet=alphabet,
    backgrounds=backgrounds,
    fonts=fonts,
    recognizer_basepath=recognizer_basepath,
    pretrained_model=None  # or 'models/recognizer_dimensions.h5'
)
```

## Project Structure

```text
eDOCr/
├── keras_ocr/          # Core OCR engine
├── keras_ocr_models/   # Model training and resources
│   ├── backgrounds/    # Training background images
│   ├── fonts/         # Engineering fonts
│   └── models/        # Trained model files
└── tools/             # Processing utilities
    ├── box_tree.py    # Rectangle detection
    ├── img_process.py # Image preprocessing
    ├── pipeline_*.py  # Specialized pipelines
    └── watermark.py   # Watermark removal
```

## Documentation

For detailed implementation insights, refer to our research paper: [Optical character recognition on engineering drawings to achieve automation in production quality control](https://www.frontiersin.org/articles/10.3389/fmtec.2023.1154132/full).

## License

[MIT License](LICENSE)