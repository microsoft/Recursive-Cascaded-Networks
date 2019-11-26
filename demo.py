import argparse
import os
import sys
import json
import re
import numpy as np

import SimpleITK as sitk
import scipy
import skimage.io
import skimage.exposure
from skimage import measure, filters, morphology
import concurrent.futures
import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-f', '--fixed', type=str, default=None,
                    help='Specifies the fixed image')
parser.add_argument('-m', '--moving', type=str, default=None,
                    help='Specifies the moving image')
parser.add_argument('-o', '--output', type=str, default='output',
                    help='Specifies the output directory')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--net_args', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    try:
        os.makedirs(args.output)
    except:
        pass
    
    img_fixed, reader_fixed = preprocess_dcm(args.fixed)
    show_image(img_fixed, os.path.join(args.output, 'fixed.png'))
    save_dcm(img_fixed, reader_fixed, os.path.join(args.output, 'fixed'))
    
    img_moving, reader_moving = preprocess_dcm(args.moving)
    show_image(img_moving, os.path.join(args.output, 'moving.png'))
    save_dcm(img_moving, reader_moving, os.path.join(args.output, 'moving'))

    import tensorflow as tf
    import tflearn

    import network

    assert args.checkpoint is not None, 'Checkpoint must be specified!'
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=[128, 128, 128], segmentation_class_value=None,
        fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    sess = tf.Session()

    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    checkpoint = args.checkpoint
    saver.restore(sess, checkpoint)
    tflearn.is_training(False, session=sess)

    keys = sum([['real_flow_{}'.format(i), 'warped_moving_{}'.format(i)] for i in range(len(framework.network.stems))], [])
    gen = [{'id1': np.ones((1,)), 'id2': np.ones((1,)),
        'voxel1': np.reshape(img_fixed, [1, 128, 128, 128, 1]), 'voxel2': np.reshape(img_moving, [1, 128, 128, 128, 1])}]
    results = framework.validate(sess, gen, keys=keys, summary=False)

    for key in keys:
        if 'flow' in key:
            im_flow = RenderFlow(results[key][0])
            skimage.io.imsave(os.path.join(args.output, key.replace('real_flow', 'flow') + '.png'), im_flow)
        else:
            warped_img = np.squeeze(results[key][0] * 255, -1).astype(np.uint8)
            show_image(warped_img, os.path.join(args.output, key + '.png'))
            save_dcm(warped_img, reader_moving, os.path.join(args.output, key))


def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


def RenderFlow(flow, coef = 15, channel = (0, 1, 2), thresh = 1):
    flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis = -1)
    #im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    #im_flow = 1 - im_flow / 20
    return im_flow

def save_dcm(img, series_reader, fpath):
    try:
        os.makedirs(fpath)
    except:
        pass
    img = img[::-1, :, ::-1]
    img = np.transpose(img, (2, 1, 0))
    filtered_image = sitk.GetImageFromArray(img)
    
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()
    
    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = filtered_image.GetDirection()
    series_tag_values = [(k, series_reader.GetMetaData(0,k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0,k)] + \
                    [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", series_reader.GetMetaData(0,"0008|103e") + " Processed-SimpleITK")] # Series Description

    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,filtered_image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(fpath, str(i) + '.dcm'))
        writer.Execute(image_slice)

def preprocess_dcm(fpath):
    img, reader = load_dcm(fpath)
    img = np.transpose(img, (2, 1, 0))
    img = img[::-1, :, ::-1]
    liver_mask = auto_liver_mask(img)
    img = crop_mask(img, liver_mask)
    return img, reader

def load_dcm(fpath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(fpath)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z, y, x
    return image_array, reader

def auto_liver_mask(vol, ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)]):
    vol = filters.gaussian(vol, sigma = 2, preserve_range = True)
    mask = np.zeros_like(vol, dtype = np.bool)
    max_area = 0
    for th_lo, th_hi in ths:
        print(th_lo, th_hi)
        bw = np.ones_like(vol, dtype = np.bool)
        bw[vol < th_lo] = 0
        bw[vol > th_hi] = 0
        if np.sum(bw) <= max_area:
            continue
        with concurrent.futures.ProcessPoolExecutor(8) as executor:
            jobs = list(range(bw.shape[-1]))
            args1 = [bw[:, :, z] for z in jobs]
            args2 = [morphology.disk(35) for z in jobs]
            for idx, ret in tqdm.tqdm(zip(jobs, executor.map(filters.median, args1, args2)), total = len(jobs)):
                bw[:, :, jobs[idx]] = ret
        # for z in range(bw.shape[-1]):
        #     bw[:, :, z] = filters.median(bw[:, :, z], morphology.disk(35))
        if np.sum(bw) <= max_area:
            continue
        labeled_seg = measure.label(bw, connectivity=1)
        regions = measure.regionprops(labeled_seg)
        max_region = max(regions, key = lambda x: x.area)
        if max_region.area <= max_area:
            continue
        max_area = max_region.area
        mask = labeled_seg == max_region.label
    assert max_area > 0, 'Failed to find the liver area!'
    return mask

def crop(arr, bound_l, bound_r, target_shape, order=1):
    cropped = arr[bound_l[0]: bound_r[0], bound_l[1]: bound_r[1], bound_l[2]: bound_r[2]]
    return scipy.ndimage.zoom(cropped, np.array(target_shape) / np.array(cropped.shape), order = order)

def wl_normalization(img, w=290, l=120):
    img = skimage.exposure.rescale_intensity(img, in_range=(l - w / 2, l + w / 2), out_range=(0, 255))
    return img.astype(np.uint8)

def crop_mask(volume, segmentation, target_shape=(128, 128, 128)):
    indices = np.array(np.nonzero(segmentation))
    bound_r = np.max(indices, axis=-1)
    bound_l = np.min(indices, axis=-1)
    box_size = bound_r - bound_l + 1
    padding = np.maximum( (box_size * 0.1).astype(np.int32), 5)
    bound_l = np.maximum(bound_l - padding, 0)
    bound_r = np.minimum(bound_r + padding + 1, segmentation.shape)
    return wl_normalization(crop(volume, bound_l, bound_r, target_shape)).astype(np.uint8)

def show_image(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    if len(imgs.shape) == 3:
        if not norm:
            if np.max(imgs) < 5:
                imgs = imgs * 255.0
            imgs = np.array(imgs, dtype=np.uint8)
    if transpose == 'z':
        if len(imgs.shape) == 3:
            imgs = np.transpose(imgs, (2, 0, 1))
        else:
            imgs = np.transpose(imgs, (2, 0, 1, 3))
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i, ax in zip(range(0, imgs.shape[0], imgs.shape[0] // 16), axes):
        if len(imgs.shape) == 4:
            ax.imshow(imgs[i], aspect='equal', origin=origin)
        elif norm:
            ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, aspect='equal', origin=origin)
        else:
            ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), norm = matplotlib.colors.NoNorm(), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig


if __name__ == '__main__':
    main()
