import json
import pdb
import numpy as np
from omegaconf import DictConfig
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys

from utils.anim import bvh, quat, txform
from utils.anim.utils import write_bvh
import torch
from scipy.signal import savgol_filter
import utils.rotation_conversions as rc



def rebuild_bone_hierarchy(original_names, original_parents, select_bone_names):
    """
    Extract a new parent-child hierarchy for selected bones from the original bone hierarchy.
    
    Args:
        original_names: List of original bone names.
        original_parents: Original parent-child relationship list (indices).
        select_bone_names: List of bone names to extract.
    
    Returns:
        new_parents: New parent-child relationship list.
    """
    # Create a mapping from original names to indices.
    name_to_index = {name: i for i, name in enumerate(original_names)}
    
    # Create a mapping from selected bone names to their new indices.
    select_name_to_new_index = {name: i for i, name in enumerate(select_bone_names)}
    
    # Initialize the new parent-child relationship list.
    new_parents = []
    
    # Find the parent bone in the new hierarchy for each selected bone.
    for bone_name in select_bone_names:
        # Get the current bone's index in the original list.
        original_index = name_to_index[bone_name]
        
        # Get the current bone's parent index in the original hierarchy.
        original_parent_index = original_parents[original_index]
        
        # If there is no parent bone (root bone), set it to -1.
        if original_parent_index == -1:
            new_parents.append(-1)
            continue
        
        # Get the original parent bone's name.
        original_parent_name = original_names[original_parent_index]
        
        # Find the nearest ancestor bone that is included in the selected list.
        current_parent_index = original_parent_index
        while current_parent_index != -1:
            current_parent_name = original_names[current_parent_index]
            
            # If the current parent bone is in the selected list, use it.
            if current_parent_name in select_name_to_new_index:
                new_parent_index = select_name_to_new_index[current_parent_name]
                new_parents.append(new_parent_index)
                break
            
            # Otherwise, continue searching upward.
            current_parent_index = original_parents[current_parent_index]
        else:
            # If no suitable parent bone is found, set it to -1 (make it a root bone).
            new_parents.append(-1)
    
    return new_parents


def forward_kinematics(joint_rotations, rest_pose_skeletion, parents):
    # Inputs
    # joint_rotations: local-space rotation matrices for each joint [bs*seq, 75, 3, 3]
    # rest_pose_skeletion: bone offsets in the rest pose [75, 3]
    # parents: parent index list for joints [75]
    
    # Outputs
    # Joint positions in character space. Add the root joint translation to obtain world-space positions.
    
    batch_size = joint_rotations.shape[0]
    joints_num = joint_rotations.shape[1]
    
    if not isinstance(rest_pose_skeletion, torch.Tensor):
        rest_pose_skeletion = torch.tensor(rest_pose_skeletion, dtype=joint_rotations.dtype, device=joint_rotations.device)
    
    global_rotations = torch.zeros_like(joint_rotations)
    global_positions = torch.zeros((batch_size, joints_num, 3), dtype=joint_rotations.dtype, device=joint_rotations.device)
    
    for i in range(joints_num):
        parent = parents[i]
        
        if parent == -1:
            global_rotations[:, i] = joint_rotations[:, i]
            global_positions[:, i] = rest_pose_skeletion[i].expand(batch_size, 3)
        else:
            global_rotations[:, i] = torch.bmm(
                global_rotations[:, parent],
                joint_rotations[:, i]
            )
            
            rotated_position = torch.matmul(
                global_rotations[:, parent],
                rest_pose_skeletion[i].unsqueeze(-1)
            ).squeeze(-1)
            
            global_positions[:, i] = global_positions[:, parent] + rotated_position
            
    return global_positions

select_bone_names = "Root_M Hip_R HipPart1_R Knee_R KneePart1_R Ankle_R Toes_R ToesEnd_R Heel_R HeelEnd_R Spine1_M Spine1Part1_M Chest_M Scapula_R Shoulder_R ShoulderPart1_R Elbow_R ElbowPart1_R Wrist_R MiddleFinger1_R MiddleFinger2_R MiddleFinger3_R MiddleFinger4_R ThumbFinger1_R ThumbFinger2_R ThumbFinger3_R ThumbFinger4_R IndexFinger1_R IndexFinger2_R IndexFinger3_R IndexFinger4_R Cup_R PinkyFinger1_R PinkyFinger2_R PinkyFinger3_R PinkyFinger4_R RingFinger1_R RingFinger2_R RingFinger3_R RingFinger4_R Neck_M NeckPart1_M Head_M Head_angleFix L_eye_jnt L_eScale_jnt Eye_L L_pupil_jnt R_eye_jnt R_eScale_jnt Eye_R R_pupil_jnt Scapula_L Shoulder_L ShoulderPart1_L Elbow_L ElbowPart1_L Wrist_L MiddleFinger1_L MiddleFinger2_L MiddleFinger3_L MiddleFinger4_L ThumbFinger1_L ThumbFinger2_L ThumbFinger3_L ThumbFinger4_L IndexFinger1_L IndexFinger2_L IndexFinger3_L IndexFinger4_L Cup_L PinkyFinger1_L PinkyFinger2_L PinkyFinger3_L PinkyFinger4_L RingFinger1_L RingFinger2_L RingFinger3_L RingFinger4_L Hip_L HipPart1_L Knee_L KneePart1_L Ankle_L Toes_L ToesEnd_L Heel_L HeelEnd_L".split()
default_meta_info_path = './stats/zm_meta_info.npz'


lower_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 79, 80, 81, 82, 83, 84, 85, 86, 87]
upper_body_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
hands_body_indices = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]


lower_body_indices_feature_indices = [] # r_velocity and xy vel and z trans
upper_body_indices_feature_indices = []
hands_body_indices_feature_indices = []
for idx in lower_body_indices:
    start = idx * 6
    end = start + 6
    lower_body_indices_feature_indices.extend(list(range(start, end)))
lower_body_indices_feature_indices.extend([528,529,530])
for idx in upper_body_indices:
    start = idx * 6
    end = start + 6
    upper_body_indices_feature_indices.extend(list(range(start, end)))
for idx in hands_body_indices:
    start = idx * 6
    end = start + 6
    hands_body_indices_feature_indices.extend(list(range(start, end)))

def get_filter_joint(anim_data):
    nframes = len(anim_data["rotations"])
    str2index = {name: i for i, name in enumerate(anim_data['names'])}
    tgt_index = [str2index[name] for name in select_bone_names if name in str2index]
    new_parents = rebuild_bone_hierarchy(anim_data['names'], anim_data['parents'], select_bone_names)
    new_rotations = anim_data["rotations"][:,tgt_index]
    new_offsets = anim_data["offsets"][tgt_index]
    new_positions = anim_data["positions"][:,tgt_index]
    
    
    anim_data["rotations"] = new_rotations
    anim_data["positions"] = new_positions
    anim_data["offsets"] = new_offsets
    anim_data["parents"] = new_parents
    anim_data["names"] = select_bone_names
    return anim_data

def preprocess_animation(animation_file, fps=60,has_joint_filter=True):
    anim_data = bvh.load(animation_file)       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    nframes = len(anim_data["rotations"])

    if has_joint_filter is False:
        anim_data = get_filter_joint(anim_data)
    
    if os.path.exists(default_meta_info_path) is False:
        np.savez(default_meta_info_path, names=anim_data["names"], parents=anim_data["parents"], offsets=anim_data["offsets"])

    if fps != 60 :
        rate = 60 // fps
        anim_data["rotations"] = anim_data["rotations"][0:nframes:rate]
        anim_data["positions"] = anim_data["positions"][0:nframes:rate]
        dt = 1 / fps
        nframes = anim_data["positions"].shape[0]
    else:
        dt = anim_data["frametime"]

    njoints = len(anim_data["parents"])


    lrot_mat = quat.to_xform(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))
    
    lrot_mat = torch.tensor(lrot_mat).float().cuda()
    
    lpos = forward_kinematics(lrot_mat, anim_data["offsets"], anim_data["parents"])
    lpos = lpos.detach().cpu().numpy()
    
    pos = lpos + anim_data["positions"][:,0:1] - anim_data["positions"][0,0:1]
    
    pos_x = pos[:, 0, 0:1]
    pos_y = pos[:, 0, 1:2]
    pos_z = pos[:, 0, 2:3]
    
    pos_vx = np.zeros_like(pos_x)
    pos_vy = np.zeros_like(pos_y)
    pos_vz = np.zeros_like(pos_z)
    
    pos_vx[1:] = (pos_x[1:] - pos_x[:-1])
    pos_vy[1:] = (pos_y[1:] - pos_y[:-1])

    
    lrot_6d = rc.matrix_to_rotation_6d(lrot_mat).reshape(nframes, njoints * 6).detach().cpu().numpy()

    repre = np.concatenate([lrot_6d, pos_vx, pos_vy, pos_z], axis=-1)
    return repre, anim_data["parents"], dt, anim_data["order"], njoints



def ensure_euler_continuity(euler_angles, threshold=180):
    """
    Fix discontinuities in Euler angles.
    When a jump larger than threshold is detected, add or subtract 360 degrees to keep it continuous.
    This preserves accumulated rotation and avoids jumps such as 175 degrees to -175 degrees.
    
    Args:
        euler_angles: Euler angle array with shape (frames, joints, 3).
        threshold: Jump threshold, default is 180 degrees.
    Returns:
        Corrected Euler angle array.
    """
    result = euler_angles.copy()
    for joint_idx in range(result.shape[1]):
        for axis in range(3):
            for i in range(1, len(result)):
                diff = result[i, joint_idx, axis] - result[i-1, joint_idx, axis]
                if diff > threshold:
                    result[i:, joint_idx, axis] -= 360
                elif diff < -threshold:
                    result[i:, joint_idx, axis] += 360
    return result


def smooth_motion_trajectory(motion_repre, window_size=5, detect_threshold=30):
    """
    Smooth the motion trajectory to fix discontinuous jumps that may appear in GPT-generated results.
    Only locally smooth regions where jumps are detected, leaving other regions unchanged.
    
    Args:
        motion_repre: Motion representation with shape (frames, 531).
        window_size: Smoothing window size.
        detect_threshold: Jump detection threshold (6D rotation change magnitude).
    Returns:
        Smoothed motion representation.
    """
    result = motion_repre.copy()
    
    # Process only the rotation part (first 528 dimensions).
    rot_6d = result[:, :528].reshape(-1, 88, 6)
    
    # Measure frame-to-frame change magnitude.
    rot_diff = np.diff(rot_6d, axis=0)
    rot_diff_norm = np.linalg.norm(rot_diff, axis=(1, 2))
    
    # Find jump positions.
    jump_frames = np.where(rot_diff_norm > detect_threshold)[0]
    
    if len(jump_frames) == 0:
        return result
    
    # Apply local smoothing around jump positions.
    for jump_frame in jump_frames:
        start = max(0, jump_frame - window_size)
        end = min(len(result), jump_frame + window_size + 1)
        
        # Smooth with a Savitzky-Golay filter.
        window = min(window_size * 2 + 1, end - start)
        if window >= 3 and window % 2 == 1:
            for joint in range(88):
                for dim in range(6):
                    result[start:end, joint * 6 + dim] = savgol_filter(
                        result[start:end, joint * 6 + dim], 
                        window_length=window, 
                        polyorder=min(2, window - 1)
                    )
    
    return result


def repre_to_bvh(tmp_bvh, filename):
    
    tmp_bvh_rot = tmp_bvh[...,:528]
    tmp_bvh_pos = tmp_bvh[...,528:]
    tmp_bvh_rot = torch.tensor(tmp_bvh_rot)
    #tmp_bvh_rot_mat = rc.rotation_6d_to_matrix(tmp_bvh_rot.reshape(-1, 88, 6)).cpu().numpy()
    #V_lrot = quat.from_xform(tmp_bvh_rot_mat)

    tmp_bvh_rot_mat = rc.rotation_6d_to_matrix(tmp_bvh_rot.reshape(-1, 88, 6))
    
    V_lrot = rc.matrix_to_quaternion(tmp_bvh_rot_mat).cpu().numpy()
    

    V_lrot = quat.unroll(V_lrot)
    
    tmp_bvh_pos = torch.tensor(tmp_bvh_pos)
    tmp_bvh_pos[...,0] = torch.cumsum(tmp_bvh_pos[...,0], dim=0)
    tmp_bvh_pos[...,1] = torch.cumsum(tmp_bvh_pos[...,1], dim=0)
    V_lpos = tmp_bvh_pos.reshape(-1, 1, 3).cpu().numpy()
    
    default_meta_info = np.load(default_meta_info_path)
    order = 'zyx'
    dt = 1 / 60
    
    # Convert to Euler angles.
    V_euler = np.degrees(quat.to_euler(V_lrot, order=order))
    
    # Fix 2: ensure Euler angle continuity (avoid +/-180 degree boundary jumps).
    V_euler = ensure_euler_continuity(V_euler)
    
    bvh.save(
    filename,
        dict(
            order=order,
            offsets=default_meta_info['offsets'],
            names=default_meta_info['names'],
            frametime=dt,
            parents=default_meta_info['parents'],
            positions=V_lpos,
            rotations=V_euler,
        ),
    )



import glob
from tqdm import tqdm


def make_zeggs_dataset(source_path,fps=60,has_joint_filter = True):

    all_poses = []
    bvh_files = sorted(glob.glob(source_path + "/*.bvh"))
    for bvh_file in tqdm(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        pose, parents, dt, order, njoints = preprocess_animation(bvh_file,fps,has_joint_filter)
        all_poses.append({'file': name,'pose': pose})
    return all_poses


bone_names = ['Global', 'Root_M', 'Hip_R', 'HipPart1_R', 'Knee_R', 'KneePart1_R', 'Ankle_R', 'Toes_R', 'ToesEnd_R', 'Toes_R_scale', 'Heel_R', 'HeelEnd_R', 'Heel_R_scale', 'Ankle_R_scale', 'KneePart1_R_scale', 'Knee_R_scale', 'HipPart1_R_scale', 'Hip_R_scale', 'Hip_L', 'HipPart1_L', 'Knee_L', 'KneePart1_L', 'Ankle_L', 'Toes_L', 'ToesEnd_L', 'Toes_L_scale', 'Heel_L', 'HeelEnd_L', 'Heel_L_scale', 'Ankle_L_scale', 'KneePart1_L_scale', 'Knee_L_scale', 'HipPart1_L_scale', 'Hip_L_scale', 'Root_M_scale', 'Spine1_M', 'Spine1_M_scale', 'Spine1Part1_M', 'Spine1Part1_M_scale', 'Chest_M', 'Scapula_L', 'Shoulder_L', 'ShoulderPart1_L', 'Elbow_L', 'ElbowPart1_L', 'Wrist_L', 'MiddleFinger1_L', 'MiddleFinger2_L', 'MiddleFinger3_L', 'MiddleFinger4_L', 'MiddleFinger3_L_scale', 'MiddleFinger2_L_scale', 'MiddleFinger1_L_scale', 'ThumbFinger1_L', 'ThumbFinger2_L', 'ThumbFinger3_L', 'ThumbFinger4_L', 'ThumbFinger3_L_scale', 'ThumbFinger2_L_scale', 'ThumbFinger1_L_scale', 'IndexFinger1_L', 'IndexFinger2_L', 'IndexFinger3_L', 'IndexFinger4_L', 'IndexFinger3_L_scale', 'IndexFinger2_L_scale', 'IndexFinger1_L_scale', 'Cup_L', 'PinkyFinger1_L', 'PinkyFinger2_L', 'PinkyFinger3_L', 'PinkyFinger4_L', 'PinkyFinger3_L_scale', 'PinkyFinger2_L_scale', 'PinkyFinger1_L_scale', 'RingFinger1_L', 'RingFinger2_L', 'RingFinger3_L', 'RingFinger4_L', 'RingFinger3_L_scale', 'RingFinger2_L_scale', 'RingFinger1_L_scale', 'Cup_L_scale', 'Wrist_L_scale', 'ElbowPart1_L_scale', 'Elbow_L_scale', 'ShoulderPart1_L_scale', 'Shoulder_L_scale', 'Scapula_L_scale', 'Chest_M_scale', 'Scapula_R', 'Shoulder_R', 'ShoulderPart1_R', 'Elbow_R', 'ElbowPart1_R', 'Wrist_R', 'MiddleFinger1_R', 'MiddleFinger2_R', 'MiddleFinger3_R', 'MiddleFinger4_R', 'MiddleFinger3_R_scale', 'MiddleFinger2_R_scale', 'MiddleFinger1_R_scale', 'ThumbFinger1_R', 'ThumbFinger2_R', 'ThumbFinger3_R', 'ThumbFinger4_R', 'ThumbFinger3_R_scale', 'ThumbFinger2_R_scale', 'ThumbFinger1_R_scale', 'IndexFinger1_R', 'IndexFinger2_R', 'IndexFinger3_R', 'IndexFinger4_R', 'IndexFinger3_R_scale', 'IndexFinger2_R_scale', 'IndexFinger1_R_scale', 'Cup_R', 'PinkyFinger1_R', 'PinkyFinger2_R', 'PinkyFinger3_R', 'PinkyFinger4_R', 'PinkyFinger3_R_scale', 'PinkyFinger2_R_scale', 'PinkyFinger1_R_scale', 'RingFinger1_R', 'RingFinger2_R', 'RingFinger3_R', 'RingFinger4_R', 'RingFinger3_R_scale', 'RingFinger2_R_scale', 'RingFinger1_R_scale', 'Cup_R_scale', 'Wrist_R_scale', 'ElbowPart1_R_scale', 'Elbow_R_scale', 'ShoulderPart1_R_scale', 'Shoulder_R_scale', 'Scapula_R_scale', 'Neck_M', 'Neck_M_scale', 'NeckPart1_M', 'NeckPart1_M_scale', 'Head_M', 'Head_angleFix', 'L_eye_jnt', 'L_eScale_jnt', 'Eye_L', 'L_pupil_jnt', 'R_eye_jnt', 'R_eScale_jnt', 'Eye_R', 'R_pupil_jnt', 'dy_jnt', 'dy_S_jnt', 'L_jnt1', 'L_jnt2', 'R_jnt1', 'R_jnt2', 'dy_B_jnt', 'B_jnt3', 'B_jnt4', 'B_jnt5', 'B_jnt6', 'B_jnt7', 'B_jnt8', 'B_jnt9', 'B_jnt10', 'B_jnt11', 'B_jnt12', 'B_jnt1', 'B_jnt2']

if __name__ == '__main__':
    
    make_dataset = True
    fps = 30
    # Whether unused BVH joints have already been filtered.
    has_joint_filter = True
    
    test_repre = False
    if test_repre:
        tmp_bvh,_,_,_,_ = preprocess_animation('./datasets/xx_zm_reduce/all/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh')
        tmp_bvh_rot = tmp_bvh[...,:528]
        tmp_bvh_pos = tmp_bvh[...,528:]
        tmp_bvh_rot = torch.tensor(tmp_bvh_rot)
        tmp_bvh_rot_mat = rc.rotation_6d_to_matrix(tmp_bvh_rot.reshape(-1, 88, 6)).cpu().numpy()
        
        V_lrot = quat.from_xform(tmp_bvh_rot_mat)
        tmp_bvh_pos = torch.tensor(tmp_bvh_pos)
        tmp_bvh_pos[...,0] = torch.cumsum(tmp_bvh_pos[...,0], dim=0)
        tmp_bvh_pos[...,1] = torch.cumsum(tmp_bvh_pos[...,1], dim=0)
        V_lpos = tmp_bvh_pos.reshape(-1, 1, 3).cpu().numpy()
        
        default_meta_info = np.load(default_meta_info_path)
        order = 'zyx'
        dt = 1 / 60
        filename ='test.bvh'
        bvh.save(
        filename,
            dict(
                order=order,
                offsets=default_meta_info['offsets'],
                names=default_meta_info['names'],
                frametime=dt,
                parents=default_meta_info['parents'],
                positions=V_lpos,
                rotations=np.degrees(quat.to_euler(V_lrot, order=order)),
            ),
        )



    
    if make_dataset and fps == 60:
        train_source_path = './datasets/xx_zm_reduce//train'
        target_path = './datasets/xx_zm_reduce/train_processed_60fps'
        train_list =  make_zeggs_dataset(train_source_path,fps=60)
        np.save(target_path, train_list)
        
        valid_source_path = './datasets/xx_zm_reduce//valid'
        target_path = './datasets/xx_zm_reduce/valid_processed_60fps'
        valid_list =  make_zeggs_dataset(valid_source_path,fps=60)
        np.save(target_path, valid_list)
        
        all_pose = []
        for item in train_list+valid_list:
            all_pose.append(item['pose'])
        all_pose = np.concatenate(all_pose,axis=0)
        mean = all_pose.mean(axis=0)
        std = all_pose.std(axis=0)
        np.save('./datasets/xx_zm_reduce/mean_std_60fps.npy', [mean, std])
        

    if make_dataset and fps == 30:
        train_source_path = './datasets/xx_zm_reduce//train'
        target_path = './datasets/xx_zm_reduce/train_processed_30fps'
        train_list =  make_zeggs_dataset(train_source_path,fps=30)
        np.save(target_path, train_list)
        
        valid_source_path = './datasets/xx_zm_reduce//valid'
        target_path = './datasets/xx_zm_reduce/valid_processed_30fps'
        valid_list =  make_zeggs_dataset(valid_source_path,fps=30)
        np.save(target_path, valid_list)
        
        all_pose = []
        for item in train_list+valid_list:
            all_pose.append(item['pose'])
        all_pose = np.concatenate(all_pose,axis=0)
        mean = all_pose.mean(axis=0)
        std = all_pose.std(axis=0)
        np.save('./datasets/xx_zm_reduce/mean_std_30fps.npy', [mean, std])
