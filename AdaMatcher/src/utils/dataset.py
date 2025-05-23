import io

import cv2
import h5py
import numpy as np
import torch
from loguru import logger
from numpy.linalg import inv
import kornia.geometry.transform as KT

try:
    # for internal use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---


def load_array_from_s3(
    path,
    client,
    cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f'==> Data loading failure: {path}')
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def imread_color(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_COLOR
    # if str(path).startswith('s3://'):
    #     image = load_array_from_s3(str(path), client, cv_type)
    # else:
    #     image = cv2.imread(str(path), cv_type)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment_fn is not None:
        image = augment_fn(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]), f'{pad_size} < {max(inp.shape[-2:])}'
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[2], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[0], :inp.shape[1]] = inp.transpose(2, 0, 1)
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---


def read_megadepth_gray(path,
                        resize=None,
                        df=None,
                        padding=False,
                        augment_fn=None,
                        rotation=0):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()
        if rotation % 2:
            scales = scales[::-1]
    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = (torch.from_numpy(image).float()[None] / 255
             )  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_color(path,
                         resize=None,
                         df=None,
                         padding=False,
                         augment_fn=None,
                         rotation=0,
                         hflip=False,
                         vflip=False):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_color(path, augment_fn, client=MEGADEPTH_CLIENT)
    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()

    if hflip:
        image = KT.hflip(torch.from_numpy(image).unsqueeze(0)).squeeze(0).numpy()

    if vflip:
        image = KT.vflip(torch.from_numpy(image).unsqueeze(0)).squeeze(0).numpy()

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    scale_wh = torch.tensor([w_new, h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = (torch.from_numpy(image).float() / 255
             )  # (3, h, w) -> (3, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale, scale_wh

def read_walkdepth_depthmap(path):
    """
    Reads WalkDepth dataset depth map binary file.
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def read_megadepth_depth(path, pad_to=None, hflip=False, vflip=False):
    if str(path).endswith('.jpg'):
        depth = cv2.imread(path, 0)
    elif str(path).endswith('.bin'):
        depth = read_walkdepth_depthmap(path)
    elif str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])

    if hflip:
        depth = KT.hflip(torch.from_numpy(depth).unsqueeze(0)).squeeze(0).numpy()

    if vflip:
        depth = KT.vflip(torch.from_numpy(depth).unsqueeze(0)).squeeze(0).numpy()

    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---
def read_scannet_color(path,
                       resize=(640, 480),
                       augment_fn=None,
                       rotation=0,
                       ret_mask=False):
    # read image
    image = imread_color(path, augment_fn)
    w, h = image.shape[1], image.shape[0]
    image = cv2.resize(image, resize)
    w_new, h_new = image.shape[1], image.shape[0]
    # image = image.transpose(2, 0, 1)
    # print(image.shape)
    # normalized

    # if rotation != 0:
    #     image = np.rot90(image, k=rotation).copy()
    # if rotation % 2:
    #     scales = scales[::-1]

    pad_to = max(resize[0], resize[1])
    if ret_mask:
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)
    else:
        image = image.transpose(2, 0, 1)
        mask = None
    image = torch.from_numpy(image).float() / 255

    # scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
    scale = torch.tensor([1.0, 1.0], dtype=torch.float)
    scale_wh = torch.tensor([w_new, h_new], dtype=torch.float)
    return image, mask, scale, scale_wh
    # return image, None, scale, scale_wh


def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_depth(path):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT,
                                   cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth, _ = pad_bottom_right(depth, 2000, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)

    return depth


def read_scannet_pose(path):
    """Read ScanNet's Camera2World pose and transform it to World2Camera.

    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """Read ScanNet's intrinsic matrix and return the 3x3 matrix."""
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]
