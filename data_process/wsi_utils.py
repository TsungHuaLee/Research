# +
import pyvips
import numpy as np
import openslide
import os

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

#map np dtypes to vips
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
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def vips_get_image(f, discard_alpha=True, cvt_npy=True):
    vips_image = pyvips.Image.new_from_file(f, access='sequential') 
    if discard_alpha:
        vips_image = vips_image.extract_band(0, n=3)
    if cvt_npy:
        vips_image = vips2numpy(vips_image)
    return vips_image 

def vips_get_wsi(slide_path: str, level=0):
    """
    Fetches the level of image from a slide path.
    Color Space in slide is RGBA. We only retrieve RGB.
    
    Args:
        slide_path:  Filename to load from.
        keys: Load this level from the file

    Returns:
         VipsImage object

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    assert os.path.exists(slide_path), "{} not exists".format(slide_path)
    vips_image = pyvips.Image.openslideload(slide_path, level=level)
    # drop the alpha channel
    vips_image = vips_image.extract_band(0, n=3)
    return vips_image

def vips_resize(vips_image, edge_resize_factor):
    vips_image = vips_image.resize(float(1/edge_resize_factor))
    return vips_image
    
def openslide_get_properties(slide_path):
    wsi = openslide.OpenSlide(slide_path)
    mppX = float(wsi.properties["openslide.mpp-x"])
    mppY = float(wsi.properties["openslide.mpp-y"])
    (width, height) = wsi.dimensions
    levelCount = int(wsi.properties["openslide.level-count"])
    downsampleFactors = np.array(wsi.level_downsamples).astype('float')
    objectivePower = int(wsi.properties["openslide.objective-power"])
    mag = objectivePower/downsampleFactors
    wsi.close()
    return mppX, mppY, width, height, levelCount, downsampleFactors, mag
