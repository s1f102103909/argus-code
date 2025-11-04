import pathlib
import tempfile
import sys
import shutil
import json
import torch
import numpy as np
import os
import cv2
from triton import TritonError
from torchvision.transforms import functional as F
from typing import Tuple

# dust3r and mast3r
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.dust3r.utils.image import load_images
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever
from mast3r.demo import _convert_scene_output_to_glb
from mast3r.dust3r.utils.device import to_numpy
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

class Cache:
    def __init__(self, should_del=True):
        os.makedirs("./cache", exist_ok=True)
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir="./cache")
        self.cache_dir = pathlib.Path(cache_dir)
        self.should_del = should_del
    def __del__(self):
        if not self.should_del:
            return
        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir)
    def __str__(self):
        return str(self.cache_dir)

def load_model(device='cuda'):
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return model

def get_3D_model_from_scene(
    scene,
    output_path,
    min_conf_thr=1.5,
    cam_size=0.2,
    TSDF_thresh=0.0,
    as_pointcloud=True,
    mask_sky=False,
    clean_depth=True,
    transparent_cams=False,
    silent=False,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(
        output_path,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        silent=silent,
    )

def run_sparse_global_alignment(
    model,
    image_paths,
    cache_path,
    img_size=512,
    lr1=0.07,
    niter1=500,
    lr2=0.014,
    niter2=200,
    optim_level="refine",
    shared_intrinsics=True,
    matching_conf_thr=5.0,
    device=torch.device("cuda"),
    scene_graph="complete",
    retrieval_model='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
):
    
    images, width_resized = load_images(image_paths, size=img_size)

    sim_matrix = None
    if 'retrieval' in scene_graph:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(image_paths)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()
    
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    scene = sparse_global_alignment(
        image_paths,
        pairs,
        cache_path=cache_path,
        model=model,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        device=device,
        opt_depth="depth" in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
    )
    return scene, width_resized

def save_cameras(scene, output_dir: str):
    intrinsics = scene.intrinsics
    poses = scene.get_im_poses()
    # confidence_masks = scene.get_masks()
    output_path = os.path.join(output_dir, "cameras.json")

    data = {"poses": poses.tolist(), "intrinsics": intrinsics.tolist()}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def get_camera_params_from_frames(frames: torch.Tensor,
                                  shared_intrinsics=True, scene_graph='complete',
                                  img_size: int=512, # long side of the image
                                  model=None,) -> Tuple[torch.tensor, torch.tensor, float]:
    '''
    frames: torch.Tensor, in T*C*H*W format, normalized to [-1, 1]
    extract frames, and return the camera parameters
    '''
    img_paths = []
    _cache = Cache()
    cache_dir = str(_cache)

    frames = (frames + 1) / 2
    for i, frame in enumerate(frames):
        img = F.to_pil_image(frame.float()) #修正
        img.save(os.path.join(cache_dir, f"frame_{i}.png"))
        img_paths.append(os.path.join(cache_dir, f"frame_{i}.png"))

    model = load_model(device=frames.device) if model is None else model.to(frames.device)
    scene, width_resized = run_sparse_global_alignment(model, img_paths, cache_dir,
                                                       shared_intrinsics=shared_intrinsics, 
                                                       device=frames.device,
                                                       scene_graph=scene_graph, img_size=img_size)

    del _cache
    
    return scene.get_im_poses(), scene.intrinsics, width_resized

def main():
    img_path_format = "/home/rl897/saved_images/set2/perspective_frame_{}.png"
    image_paths = [img_path_format.format(i) for i in range(3)]
    
    output_dir = './cache'
    
    model = load_model()
    scene = run_sparse_global_alignment(model, image_paths, shared_intrinsics=True)

    print(scene.get_im_poses().shape, scene.intrinsics.shape)

    save_cameras(scene, output_dir)

    get_3D_model_from_scene(scene, os.path.join(output_dir, "3d_model.glb"))

if __name__ == "__main__":
    main()