# +
from PIL import Image
import pyvips
import os
import matplotlib.pyplot as plt
import numpy as np
from .wsi_utils import *
import pickle
    
# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def oldstitch(wsi_name = None, edge_resize_factor = 4, is_plot = False, tile_size = 224, overlap = 112, x_y_pairs = None, preds = None):
    if os.path.exists("/home/Tsung/pathology/data/tcga/svs/20px/{}.svs".format(wsi_name)):
        wsi_path = "/home/Tsung/pathology/data/tcga/svs/20px/{}.svs".format(wsi_name)
    elif os.path.exists("/home/Tsung/pathology/data/tcga/svs/40px/{}.svs".format(wsi_name)):
        wsi_path = "/home/Tsung/pathology/data/tcga/svs/40px/{}.svs".format(wsi_name)
    elif os.path.exists("/home/Tsung/pathology/data/tcga/svs/normal/{}.svs".format(wsi_name)):
        wsi_path = "/home/Tsung/pathology/data/tcga/svs/normal/{}.svs".format(wsi_name)
    assert os.path.exists(wsi_path), "File not exists"

    image = pyvips.Image.openslideload(wsi_path, level=0)
    
    image = image.resize(float(1/edge_resize_factor))
    if x_y_pairs is None:
        x_y_pairs = np.load("/data/tcga/224denseTumor/{}.npy".format(wsi_name))
    
    tumor_mask = np.zeros((image.height, image.width)).astype('float32')
    if preds is None:
        for x, y in x_y_pairs:
            start_x = int(x/edge_resize_factor)
            end_x = int((x+tile_size-overlap)/edge_resize_factor)
            start_y = int(y/edge_resize_factor)
            end_y = int((y+tile_size-overlap)/edge_resize_factor)
            tumor_mask[start_y:end_y, start_x:end_x] = 1
    else:
        for xy, pred in zip(x_y_pairs, preds):
            x, y = xy
            start_x = int(x/edge_resize_factor)
            end_x = int((x+tile_size-overlap)/edge_resize_factor)
            start_y = int(y/edge_resize_factor)
            end_y = int((y+tile_size-overlap)/edge_resize_factor)
            tumor_mask[start_y:end_y, start_x:end_x] = pred
    if is_plot:
        np_3d = vips2numpy(image)
        
        plt.figure(figsize=(10, 5))
        plt.title(wsi_name)
        plt.subplot(1, 2, 1)
        plt.imshow(np_3d)
        plt.subplot(1, 2, 2)
        plt.imshow(tumor_mask)
        
    return tumor_mask, (image.width, image.height)

def stitch(wsi_name = None, edge_resize_factor = 32, tile_size = 512, overlap = 256, x_y_pairs = None, preds = None, title = ""):
    ORIGINAL_FOLDER = "/data/tcga/svs/masks"
    original_image = vips_get_image(os.path.join(ORIGINAL_FOLDER, wsi_name[:-4]+".png"))
    
    with open("/data/TMB/data/svs_x_y.pkl", "rb") as fp:
        svs_x_y_dict = pickle.load(fp)
    width, height = svs_x_y_dict[wsi_name]
    tumor_mask = np.zeros((int(height/edge_resize_factor), int(width/edge_resize_factor))).astype('float32')

    for xy, pred in zip(x_y_pairs, preds):
        x, y = xy
        start_x = int(x/edge_resize_factor)
        end_x = int((x+tile_size)/edge_resize_factor)
        start_y = int(y/edge_resize_factor)
        end_y = int((y+tile_size)/edge_resize_factor)
        tumor_mask[start_y:end_y, start_x:end_x] = pred
    
    tumor_mask[tumor_mask == 0.0] = np.nan
    """ plt heatmap """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].title.set_text(title)
    im = ax[0].imshow(tumor_mask, cmap = 'coolwarm', vmin=0, vmax=1)
    cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
    plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    """ plt original image """
    im = ax[1].imshow(original_image, vmin=0, vmax=255)
    plt.show()
    
    return 
# -


