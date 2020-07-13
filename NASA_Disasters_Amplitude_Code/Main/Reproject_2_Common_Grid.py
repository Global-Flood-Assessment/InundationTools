import rasterio
import rasterio.mask
from statistics import mean
import math
import numpy as np
from rasterio import Affine as Affine
from rasterio.warp import reproject, Resampling


def determine_max_bounds(input_tifs):
    left_bound_list, bottom_bound_list, right_bound_list, top_bound_list = [], [], [], []
    for tif_file in input_tifs:
        tif_file_open = rasterio.open(tif_file)
        tif_file_bounds = tif_file_open.bounds
        left_bound, bottom_bound, right_bound, top_bound = tif_file_bounds[0], tif_file_bounds[1], tif_file_bounds[2], \
                                                           tif_file_bounds[3]
        left_bound_list.append(left_bound)
        bottom_bound_list.append(bottom_bound)
        right_bound_list.append(right_bound)
        top_bound_list.append(top_bound)
        tif_file_open.close()

    (left_bound_f, bottom_bound_f, right_bound_f, top_bound_f) = min(left_bound_list), min(bottom_bound_list), max(
        right_bound_list), max(top_bound_list)
    return (left_bound_f, bottom_bound_f, right_bound_f, top_bound_f)


def get_pixel_size(input_tifs):
    x_pixel_size_list, y_pixel_size_list = [], []
    for tif_file in input_tifs:
        tif_file_open = rasterio.open(tif_file)
        # a b c d e f: x_size, shift, left_lat, shift, y_size, top_lon
        tif_file_transform = tif_file_open.transform
        x_pixel_size, y_pixel_size = tif_file_transform[0], tif_file_transform[4]
        x_pixel_size_list.append(x_pixel_size)
        y_pixel_size_list.append(y_pixel_size)
    x_mean_pixel_size = mean(x_pixel_size_list)
    y_mean_pixel_size = mean(y_pixel_size_list)
    return (x_mean_pixel_size, y_mean_pixel_size)


def make_new_affine(left_bound_f, top_bound_f, x_pixel_size, y_pixel_size):
    New_Affine = Affine(x_pixel_size, 0.0, left_bound_f, 0.0, y_pixel_size, top_bound_f)
    return New_Affine


def find_rows_cols(left_bound_f, bottom_bound_f, right_bound_f, top_bound_f, x_pixel_size, y_pixel_size):
    cols = math.ceil(abs((right_bound_f - left_bound_f) / (x_pixel_size))) + 25
    rows = math.ceil(abs((top_bound_f - bottom_bound_f) / (y_pixel_size))) + 25
    return (cols, rows)


def rows_cols(tif_file):
    tif_open = rasterio.open(tif_file)
    rows = tif_open.height
    cols = tif_open.width
    return (cols, rows)


def make_numpy_zeros_array(cols, rows):
    projected_array = np.zeros((rows, cols)).astype('float32')
    return projected_array


def tif_2_array(tif_file):
    tif_open = rasterio.open(tif_file)
    tif_array = tif_open.read(1)
    tif_open.close()
    return tif_array


def tif_transform(tif_file):
    tif_open = rasterio.open(tif_file)
    tif_transform = tif_open.transform
    tif_open.close()
    return tif_transform


def tif_crs(tif_file):
    tif_open = rasterio.open(tif_file)
    tif_crs = tif_open.crs
    tif_open.close()
    return tif_crs


def get_kwargs_reproject(tif_file, cols, rows, New_Affine):
    tif_open = rasterio.open(tif_file)
    kwargs_tif = tif_open.meta.copy()
    kwargs_tif.update({'width': cols,
                       'height': rows,
                       'transform': New_Affine,
                       'nodata': 0,
                       'dtype': 'float32'})
    tif_open.close()
    return kwargs_tif


def get_kwargs(tif_file):
    tif_open = rasterio.open(tif_file)
    kwargs_tif = tif_open.meta.copy()
    tif_open.close()
    return kwargs_tif


def new_tif_name_transformed(tif_file_full_path):
    sep = '/'
    new_tif_name = tif_file_full_path.split('/')[-1].split('.')[0] + '_Ptf.tif'
    new_tif_name_full_path = sep.join(tif_file_full_path.split('/')[:-1]) + '/' + new_tif_name
    return new_tif_name_full_path

def new_tif_name_reproject(tif_file_full_path):
    sep = '/'
    new_tif_name = tif_file_full_path.split('/')[-1].split('.')[0] + '_common_grid.tif'
    new_tif_name_full_path = sep.join(tif_file_full_path.split('/')[:-1]) + '/' + new_tif_name
    return new_tif_name_full_path


def new_tif_name_cut(tif_file_full_path):
    sep = '/'
    new_tif_name = tif_file_full_path.split('/')[-1].split('.')[0] + '_cut.tif'
    new_tif_name_full_path = sep.join(tif_file_full_path.split('/')[:-1]) + '/' + new_tif_name
    return new_tif_name_full_path


def reproject_2_same_grid(input_tif, New_Affine, projected_array):
    Old_array = tif_2_array(input_tif)
    Old_transform = tif_transform(input_tif)
    Old_CRS = tif_crs(input_tif)
    reproject(
        Old_array, projected_array,
        src_transform=Old_transform,
        dst_transform=New_Affine,
        src_crs=Old_CRS,
        dst_crs=Old_CRS,
        resampling=Resampling.bilinear)
    return projected_array


def write_reprojected_array_2_tif(projected_array, new_tif_name_full_path, kwargs_tif):
    reprojected_tif = rasterio.open(new_tif_name_full_path, 'w', **kwargs_tif)
    reprojected_tif.write(projected_array, indexes=1)
    reprojected_tif.close()


def power_transform(input_tifs, v):
    for input_tif in input_tifs:
        new_tif_name_full_path = new_tif_name_transformed(input_tif)
        kwargs = get_kwargs(input_tif)
        tif_array = tif_2_array(input_tif)
        tif_array_transformed = tif_array**v
        write_reprojected_array_2_tif(tif_array_transformed, new_tif_name_full_path, kwargs)


def tifs_2_same_grid(input_tifs):
    """
    Reprojects all pre-processed amplitude data to the same grid
    :param input_tifs: input of all vv and vh tifs
    :return: outputs common grid tifs
    """
    (left_bound_f, bottom_bound_f, right_bound_f, top_bound_f) = determine_max_bounds(input_tifs)
    (x_mean_pixel_size, y_mean_pixel_size) = get_pixel_size(input_tifs)
    New_Affine = make_new_affine(left_bound_f, top_bound_f, x_mean_pixel_size, y_mean_pixel_size)
    (cols, rows) = find_rows_cols(left_bound_f, bottom_bound_f, right_bound_f, top_bound_f, x_mean_pixel_size,
                                  y_mean_pixel_size)
    projected_array = make_numpy_zeros_array(cols, rows)
    for input_tif in input_tifs:
        kwargs_tif = get_kwargs_reproject(input_tif, cols, rows, New_Affine)
        reprojected_array = reproject_2_same_grid(input_tif, New_Affine, projected_array)
        new_tif_name_full_path = new_tif_name_reproject(input_tif)
        write_reprojected_array_2_tif(reprojected_array, new_tif_name_full_path, kwargs_tif)


def cut_tifs(input_tifs):
    """
    Masking data so every pixel has a data point for all images, overlap in time and space
    :param input_tifs: input list of reprojected tifs
    :return: outputs masked tifs
    """
    how_many_tif = len(input_tifs)
    cols, rows = rows_cols(input_tifs[0])
    base_array = make_numpy_zeros_array(cols, rows)
    for input_tif in input_tifs:
        tif_array = tif_2_array(input_tif)
        tif_bin = np.where(tif_array != 0,
                           1,
                           0)
        base_array = base_array + tif_bin
    mask_array = np.where(base_array == how_many_tif,
                          1,
                          0)
    for input_tif in input_tifs:
        tif_array = tif_2_array(input_tif)
        tif_final_array = (tif_array*mask_array).astype('float32')
        kwargs = get_kwargs(input_tif)
        new_tif_save_name = new_tif_name_cut(input_tif)
        write_reprojected_array_2_tif(tif_final_array, new_tif_save_name, kwargs)