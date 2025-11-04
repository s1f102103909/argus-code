import os
import cv2
import numpy as np
import torch
import math
from equilib import equi2pers
from tqdm import tqdm

def pers2equi(img: torch.Tensor, fov_x: float, height: int, width: int, 
              roll: float = None, yaw: float = None, pitch: float = None,
              rots: dict = None,
              return_mask: bool = False) -> torch.Tensor:
    # yaw is left/right angle, pitch is up/down angle, roll is rotation angle, all in radians
    # fov_x in degrees
    # img: (3, H, W), in torch tensor, range [-1, 1]
    # return: (3, H, W), in torch tensor, range [-1, 1]

    if rots is not None:
        roll = rots['roll']
        yaw = rots['yaw']
        pitch = rots['pitch']
    else:
        assert roll is not None and yaw is not None and pitch is not None

    img = img.to(torch.float32).cpu().numpy().transpose(1, 2, 0)
    img = ((img + 1) * 127.5).astype(np.uint8)

    _height, _width, _ = img.shape

    fov_x = fov_x
    fov_y = 2 * np.arctan(np.tan(np.radians(fov_x) / 2) * float(_height) / _width) / np.pi * 180
    
    len_x = np.tan(np.radians(fov_x / 2.0))
    len_y = np.tan(np.radians(fov_y / 2.0))

    x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map, y_map, z_map), axis=2)

    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * yaw)
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * (-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(np.dot(R2, R1), x_axis) * (-roll))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)
    R3 = np.linalg.inv(R3)

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R3, xyz)
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([height, width, 3])
    inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

    xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    lon_map = np.where(
        (-len_x < xyz[:, :, 1])
        & (xyz[:, :, 1] < len_x)
        & (-len_y < xyz[:, :, 2])
        & (xyz[:, :, 2] < len_y),
        (xyz[:, :, 1] + len_x) / 2 / len_x * _width,
        0,
    )
    lat_map = np.where(
        (-len_x < xyz[:, :, 1])
        & (xyz[:, :, 1] < len_x)
        & (-len_y < xyz[:, :, 2])
        & (xyz[:, :, 2] < len_y),
        (-xyz[:, :, 2] + len_y) / 2 / len_y * _height,
        0,
    )
    mask = np.where(
        (-len_x < xyz[:, :, 1])
        & (xyz[:, :, 1] < len_x)
        & (-len_y < xyz[:, :, 2])
        & (xyz[:, :, 2] < len_y),
        1,
        0,
    )

    persp = cv2.remap(
        img, # (H_, W_, 3)
        lon_map.astype(np.float32), # (H, W)
        lat_map.astype(np.float32), # (H, W)
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    ) # (H, W, 3)

    mask = mask * inverse_mask
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    persp = persp * mask

    persp = torch.tensor(persp.transpose(2, 0, 1)).float() / 127.5 - 1 # back to (3, H, W), range [-1, 1]

    if return_mask:
        mask = torch.tensor(mask.transpose(2, 0, 1))[0:1].float()
        return persp, mask
    else:
        return persp
    
def generate_mask(fov_x: float, 
                  roll: float = None, yaw: float = None, pitch: float = None,
                  rots: dict = None, 
                  height: int = 256, width: int = 512,
                  fov_y: float = None, hw_ratio: float = 2/3):
    '''
    fov_x, fov_y are in degrees
    use a random image to generate mask
    height, width are the size of the final equi-rectangular image
    '''

    if rots is not None:
        roll = rots['roll']
        yaw = rots['yaw']
        pitch = rots['pitch']
    else:
        assert roll is not None and yaw is not None and pitch is not None
    
    len_x = np.tan(np.radians(fov_x / 2.0))

    fov_y = fov_y if fov_y is not None else 2 * np.arctan(np.tan(np.radians(fov_x) / 2) * hw_ratio) / np.pi * 180
    len_y = np.tan(np.radians(fov_y / 2.0))
            
    x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map, y_map, z_map), axis=2)

    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * yaw)
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * (-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(np.dot(R2, R1), x_axis) * (-roll))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)
    R3 = np.linalg.inv(R3)

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R3, xyz)
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([height, width, 3])
    inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

    xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    mask = np.where(
        (-len_x < xyz[:, :, 1])
        & (xyz[:, :, 1] < len_x)
        & (-len_y < xyz[:, :, 2])
        & (xyz[:, :, 2] < len_y),
        1,
        0,
    )

    mask = mask * inverse_mask
    return torch.tensor(mask).float().unsqueeze(0) # (1, H, W)

def rodrigues_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(rvec)
    if theta < 1e-6:
        return torch.eye(3)
    r = rvec / theta
    K = torch.tensor([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], 
                        dtype=torch.float32, device=rvec.device)
    R = torch.eye(3, device=rvec.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)
    return R

def generate_mask_batch(fov_x: float, 
                        roll: list | torch.Tensor | np.ndarray,
                        yaw: list | torch.Tensor | np.ndarray,
                        pitch: list | torch.Tensor | np.ndarray,
                        height: int = 256, 
                        width: int = 512,
                        fov_y: float = None, 
                        hw_ratio: float = 2/3,
                        device: str = 'cpu'):
    '''
    fov_x, fov_y are in degrees. Supports batch processing.
    roll, yaw, pitch are tensors of shape (B,) for batch size B
    rots is a dictionary containing tensors of shape (B,) for 'roll', 'yaw', and 'pitch'
    '''
    if isinstance(roll, list):
        roll = torch.tensor(roll, device=device).float()
        yaw = torch.tensor(yaw, device=device).float()
        pitch = torch.tensor(pitch, device=device).float()
    elif isinstance(roll, np.ndarray):
        roll = torch.tensor(roll, device=device).float()
        yaw = torch.tensor(yaw, device=device).float()
        pitch = torch.tensor(pitch, device=device).float()
    else:
        roll = roll.float().to(device)
        yaw = yaw.float().to(device)
        pitch = pitch.float().to(device)

    B = roll.shape[0]
    assert yaw.shape[0] == B and pitch.shape[0] == B
    
    len_x = np.tan(np.radians(fov_x / 2.0))
    fov_y = fov_y if fov_y is not None else 2 * np.arctan(np.tan(np.radians(fov_x) / 2) * hw_ratio) / np.pi * 180
    len_y = np.tan(np.radians(fov_y / 2.0))
    
    # Create meshgrid
    x = (torch.linspace(-180, 180, width, device=device).unsqueeze(0).expand(height, width) * (math.pi / 180))  # (H, W)
    y = (torch.linspace(90, -90, height, device=device).unsqueeze(1).expand(height, width) * (math.pi / 180))  # (H, W)

    x_map = torch.cos(x) * torch.cos(y)
    y_map = torch.sin(x) * torch.cos(y)
    z_map = torch.sin(y)
    
    xyz = torch.stack((x_map, y_map, z_map), dim=-1)  # (H, W, 3)

    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # Create rotation matrices for each image in the batch
    R1_batch, R2_batch, R3_batch = torch.empty(B, 3, 3, device=device), torch.empty(B, 3, 3, device=device), torch.empty(B, 3, 3, device=device)

    for b in range(B):
        yaw_b = yaw[b]
        pitch_b = pitch[b]
        roll_b = roll[b]

        # Rodrigues rotation matrices
        R1 = rodrigues_to_matrix(z_axis * yaw_b).to(device)
        R2 = rodrigues_to_matrix(R1 @ y_axis * (-pitch_b)).to(device)
        R3 = rodrigues_to_matrix(torch.mm(R2, R1) @ x_axis * (-roll_b)).to(device)

        R1_batch[b] = R1
        R2_batch[b] = R2
        R3_batch[b] = R3

    # reverse
    # R1_batch = torch.inverse(R1_batch)
    # R2_batch = torch.inverse(R2_batch)
    # # R3_batch = torch.inverse(R3_batch)
    R1_batch = torch.linalg.inv_ex(R1_batch)[0]
    R2_batch = torch.linalg.inv_ex(R2_batch)[0]
    R3_batch = torch.linalg.inv_ex(R3_batch)[0]

    # Apply transformations to the xyz map for each image in the batch
    xyz = xyz.view(-1, 3).T  # (3, H*W)

    xyz_transformed = torch.matmul(R3_batch, xyz) # (B, 3, H*W)
    xyz_transformed = torch.matmul(R2_batch, xyz_transformed)
    xyz_transformed = torch.matmul(R1_batch, xyz_transformed).permute(0, 2, 1)  # (B, H*W, 3)
    xyz_transformed = xyz_transformed.view(B, height, width, 3)

    inverse_mask = (xyz_transformed[:, :, :, 0] > 0).float()

    # Normalize by x-coordinate
    xyz_normalized = xyz_transformed / xyz_transformed[:, :, :, 0].unsqueeze(-1)

    mask = ((-len_x < xyz_normalized[:, :, :, 1]) & (xyz_normalized[:, :, :, 1] < len_x) &
            (-len_y < xyz_normalized[:, :, :, 2]) & (xyz_normalized[:, :, :, 2] < len_y)).float()
    
    mask = mask * inverse_mask
    return mask.unsqueeze(1)  # (B, 1, H, W)

def pers2equi_batch(img: torch.Tensor, 
                    fov_x: float, 
                    roll: list | torch.Tensor | np.ndarray,
                    yaw: list | torch.Tensor | np.ndarray,
                    pitch: list | torch.Tensor | np.ndarray,
                    height: int = 256, width: int = 512,
                    device: str = 'cpu',
                    shrink_mask_pixels: int = 0,
                    return_mask=False) -> torch.Tensor:
    '''
    img: (B, 3, H_, W_), in torch tensor, range [-1, 1]
    roll, yaw, pitch are tensors of shape (B,) for batch size B
    for squeeze mask pixels, the mask will be eroded by that many pixels, default 0
    return: (B, 3, H, W), in torch tensor, range [-1, 1]
    '''
    if isinstance(roll, list):
        roll = torch.tensor(roll).float().to(img.device)
        yaw = torch.tensor(yaw).float().to(img.device)
        pitch = torch.tensor(pitch).float().to(img.device)
    elif isinstance(roll, np.ndarray):
        roll = torch.tensor(roll).float().to(img.device)
        yaw = torch.tensor(yaw).float().to(img.device)
        pitch = torch.tensor(pitch).float().to(img.device)
    else:
        roll = roll.float().to(img.device)
        yaw = yaw.float().to(img.device)
        pitch = pitch.float().to(img.device)
    
    B = img.shape[0]
    assert yaw.shape[0] == B and pitch.shape[0] == B and roll.shape[0] == B

    img = (img + 1) / 2

    _height, _width = img.shape[2], img.shape[3]

    len_x = np.tan(np.radians(fov_x / 2.0))
    fov_y = 2 * np.arctan(np.tan(np.radians(fov_x) / 2) * _height / _width) / np.pi * 180
    len_y = np.tan(np.radians(fov_y / 2.0))

    # Create meshgrid
    x = (torch.linspace(-180, 180, width, device=device).unsqueeze(0).expand(height, width) * (math.pi / 180))  # (H, W)
    y = (torch.linspace(90, -90, height, device=device).unsqueeze(1).expand(height, width) * (math.pi / 180))  # (H, W)

    x_map = torch.cos(x) * torch.cos(y)
    y_map = torch.sin(x) * torch.cos(y)
    z_map = torch.sin(y)
    
    xyz = torch.stack((x_map, y_map, z_map), dim=-1)  # (H, W, 3)

    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # Create rotation matrices for each image in the batch
    R1_batch, R2_batch, R3_batch = torch.empty(B, 3, 3, device=device), torch.empty(B, 3, 3, device=device), torch.empty(B, 3, 3, device=device)

    for b in range(B):
        yaw_b = yaw[b]
        pitch_b = pitch[b]
        roll_b = roll[b]

        # Rodrigues rotation matrices
        R1 = rodrigues_to_matrix(z_axis * yaw_b).to(device)
        R2 = rodrigues_to_matrix(R1 @ y_axis * (-pitch_b)).to(device)
        R3 = rodrigues_to_matrix(torch.mm(R2, R1) @ x_axis * (-roll_b)).to(device)

        R1_batch[b] = R1
        R2_batch[b] = R2
        R3_batch[b] = R3

    # reverse
    # R1_batch = torch.inverse(R1_batch)
    # R2_batch = torch.inverse(R2_batch)
    # R3_batch = torch.inverse(R3_batch)
    R1_batch = torch.linalg.inv_ex(R1_batch)[0]
    R2_batch = torch.linalg.inv_ex(R2_batch)[0]
    R3_batch = torch.linalg.inv_ex(R3_batch)[0]

    # Apply transformations to the xyz map for each image in the batch
    xyz = xyz.view(-1, 3).T  # (3, H*W)

    xyz_transformed = torch.matmul(R3_batch, xyz) # (B, 3, H*W)
    xyz_transformed = torch.matmul(R2_batch, xyz_transformed)
    xyz_transformed = torch.matmul(R1_batch, xyz_transformed).permute(0, 2, 1)  # (B, H*W, 3)
    xyz_transformed = xyz_transformed.view(B, height, width, 3)

    inverse_mask = (xyz_transformed[:, :, :, 0] > 0).float()

    # Normalize by x-coordinate
    xyz_normalized = xyz_transformed / xyz_transformed[:, :, :, 0].unsqueeze(-1)

    mask = ((-len_x < xyz_normalized[:, :, :, 1]) & (xyz_normalized[:, :, :, 1] < len_x) &
            (-len_y < xyz_normalized[:, :, :, 2]) & (xyz_normalized[:, :, :, 2] < len_y)).float()

    # erode mask
    if shrink_mask_pixels > 0:
        kernel_size = 2 * shrink_mask_pixels + 1
        mask = 1 - torch.nn.functional.conv2d(1 - mask.unsqueeze(1), torch.ones(1, 1, kernel_size, kernel_size, device=device), padding=shrink_mask_pixels).clamp(0, 1).squeeze(1)
    
    lon_map = torch.where(
        (-len_x < xyz_normalized[:, :, :, 1]) & (xyz_normalized[:, :, :, 1] < len_x) & 
        (-len_y < xyz_normalized[:, :, :, 2]) & (xyz_normalized[:, :, :, 2] < len_y),
        (xyz_normalized[:, :, :, 1]) / len_x, 0,)
    
    lat_map = torch.where( 
        (-len_x < xyz_normalized[:, :, :, 1]) & (xyz_normalized[:, :, :, 1] < len_x) & 
        (-len_y < xyz_normalized[:, :, :, 2]) & (xyz_normalized[:, :, :, 2] < len_y),
        (-xyz_normalized[:, :, :, 2]) / len_y , 0,)

    # lon_map = xyz_normalized[:, :, :, 1] / len_x
    # lat_map = -xyz_normalized[:, :, :, 2] / len_y
    
    persp = torch.nn.functional.grid_sample(img, torch.stack((lon_map, lat_map), dim=-1), mode='bilinear', align_corners=False) # (B, 3, H, W)

    mask = mask * inverse_mask # (B, H, W)
    #persp = persp * mask.unsqueeze(1) # (B, 3, H, W)
    mask_b = mask.to(torch.bool).unsqueeze(1) #修正
    persp.masked_fill_(~mask_b, 0) #修正

    #persp = persp * 2 - 1
    persp.mul_(2).add_(-1)

    if return_mask:
        mask = mask.unsqueeze(1)
        return persp, mask
    else:
        return persp
    

def partial360_to_pers(video_path, save_path, fov_x=90, hw_ratio=1, 
                       height=768, width=1536, device='cpu'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = torch.tensor(frames, device=device).permute(0, 3, 1, 2).float() / 255
    height_partial, width_partial = frames.shape[2], frames.shape[3]
    # zero pad to height, width
    canvas = torch.zeros(frames.shape[0], 3, height, width).to(device)
    canvas[:, :, (height - height_partial) // 2:(height + height_partial) // 2, (width - width_partial) // 2:(width + width_partial) // 2] = frames

    pers_frames = equi2pers(canvas, fov_x=fov_x, rots=[{'roll': 0, 'yaw': 0, 'pitch': 0}] * frames.shape[0], 
                            height=int(hw_ratio * 640), width=640, z_down=True)
    pers_frames = (pers_frames * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    pers_frames = pers_frames.transpose(0, 2, 3, 1)

    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, int(hw_ratio * 640)))
    for frame in pers_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

if __name__ == '__main__':

    root_path = '/home/rl897/360VideoGeneration/experiments-svd/1103-narrow-cut/checkpoint-5500/unet/stablization'
    save_root_path = '/home/rl897/360VideoGeneration/experiments-svd/1103-narrow-cut/checkpoint-5500/unet/stablization-pers'
    os.makedirs(save_root_path, exist_ok=True)

    video_names = [x for x in os.listdir(root_path) if x.endswith('.mp4') and 'output' in x]

    for video_name in tqdm(video_names):
        fov_x = float(video_name.split('fov')[1].split('_')[0])
        hw_ratio = float(video_name.split('hw')[1].split('.mp4')[0])

        video_path = os.path.join(root_path, video_name)
        video_save_path = os.path.join(save_root_path, video_name)

        if os.path.exists(video_save_path):
            continue

        partial360_to_pers(video_path, video_save_path, fov_x=fov_x, hw_ratio=hw_ratio, height=576, width=1152, device='cuda:0')