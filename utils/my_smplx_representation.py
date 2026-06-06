import sys
sys.path.append('/mnt/data/cbh/SynTalker')
import numpy as np
from utils import quat
from utils import rotation_conversions as rc
import torch


def get_motion_representations(poses,trans):
    
    ## 这个函数最后返回：root的y轴旋转角速度(1)，root的xz平面的速度(2)，root的y轴高度(1)，以及人物的关节rot6d旋转(330)（注意根关节的y轴旋转已被分离出来了）
    
    trans[:,0] = trans[:,0] - trans[0,0]
    trans[:,2] = trans[:,2] - trans[0,2]

    ### 使得初始帧通通面向Z+轴
    root_quat = quat.from_angle_axis(poses[:,:3])

    root_fwd = quat.mul_vec(root_quat, np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]

    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )

    root_quat_1 = quat.mul(quat.inv(root_rot[0]), root_quat)
    trans_1 = quat.mul_vec(quat.inv(root_rot[0]), trans)
    root_rot_1 = quat.mul(quat.inv(root_rot[0]), root_rot)

    
    trans_1_x = trans_1[:, 0:1]
    trans_1_y = trans_1[:, 1:2]
    trans_1_z = trans_1[:, 2:3]

    trans_1_v_x = trans_1_x[1:]-trans_1_x[:-1]
    trans_1_v_z = trans_1_z[1:]-trans_1_z[:-1]

    trans_1_v_x_0 = trans_1_v_x[0] - (trans_1_v_x[2] - trans_1_v_x[1])
    trans_1_v_x = np.concatenate([trans_1_v_x_0.reshape(1, -1), trans_1_v_x], axis=0)
    
    trans_1_v_z_0 = trans_1_v_z[0] - (trans_1_v_z[2] - trans_1_v_z[1])
    trans_1_v_z = np.concatenate([trans_1_v_z_0.reshape(1, -1), trans_1_v_z], axis=0)
    
    
    trans_1_v = np.concatenate([trans_1_v_x,np.zeros_like(trans_1_v_x),trans_1_v_z], axis=-1)
    trans_1_v = quat.mul_vec(quat.inv(root_rot_1),trans_1_v)
    trans_1_v_x = trans_1_v[:, 0:1]
    trans_1_v_z = trans_1_v[:, 2:3]
    

    ### 下面开始分离出根关节的在y轴的旋转，将其以角速度表示

    # root_quat_2这个就是分理出跟关节点绕y轴旋转后的根关节点的四元数旋转
    
    r_velocity = quat.mul(root_rot_1[1:], quat.inv(root_rot_1[:-1]))
    r_velocity = 2 * np.arcsin(r_velocity[:, 2:3])
    root_quat_2 = quat.mul(quat.inv(root_rot_1), root_quat_1)
    angle_axis = quat.to_angleaxis(root_quat_2)
    poses[:, :3] = angle_axis
    poses = poses.reshape(-1, 55, 3)
    
    poses = torch.from_numpy(poses).float()
    poses_matrix = rc.axis_angle_to_matrix(poses)
    poses_rot6d = rc.matrix_to_rotation_6d(poses_matrix)
    poses_rot6d = poses_rot6d.numpy().reshape(-1, 55*6)
    
    
    r_velocity_0 = r_velocity[0] - (r_velocity[2] - r_velocity[1])
    r_velocity = np.concatenate([r_velocity_0.reshape(1, -1), r_velocity], axis=0)
    

    motion_representation = np.concatenate([r_velocity,trans_1_v_x,trans_1_v_z,trans_1_y,poses_rot6d],axis=-1)
    return motion_representation
    


def from_representations_to_motion(motion_representation,init_pose = None):
    r_velocity = motion_representation[:, :1]
    trans_1_v_x = motion_representation[:, 1:2]
    trans_1_v_z = motion_representation[:, 2:3]
    trans_1_y = motion_representation[:, 3:4]
    poses_rot6d = motion_representation[:, 4:]

    if init_pose is None:
        r_velocity[0]=0
        trans_1_v_x[0]=0
        trans_1_v_z[0]=0
    else:
        pass
    
    poses_rot6d = torch.from_numpy(poses_rot6d).float().reshape(-1, 55, 6)
    poses_matrix = rc.rotation_6d_to_matrix(poses_rot6d)
    poses_axix_angle = rc.matrix_to_axis_angle(poses_matrix)
    poses_axix_angle = poses_axix_angle.numpy().reshape(-1, 55*3)
    
    root_y_rot = np.cumsum(r_velocity, axis=0)
    r_rot_quat = np.zeros([root_y_rot.shape[0], 4])
    r_rot_quat[..., 0:1] = np.cos(root_y_rot/2)
    r_rot_quat[..., 2:3] = np.sin(root_y_rot/2)
    
    pelvis_quat = quat.from_angle_axis(poses_axix_angle[:,:3])
    
    pelvis_quat = quat.mul(r_rot_quat,pelvis_quat)
    angle_axis = quat.to_angleaxis(pelvis_quat)
    poses_axix_angle[:, :3] = angle_axis
    
    
    trans_1_v = np.concatenate([trans_1_v_x,np.zeros_like(trans_1_v_x),trans_1_v_z], axis=-1)
    trans_1_v = quat.mul_vec(r_rot_quat,trans_1_v)
    trans_1 = np.cumsum(trans_1_v, axis=0)
    
    trans_1[:,1:2]=trans_1_y

    
    return poses_axix_angle, trans_1

if __name__ == '__main__':
    # 加载数据
    npz_file_path = '/mnt/data3/cbh/SynTalker/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/smplxflame_30/2_scott_0_77_77.npz'
    a = np.load(npz_file_path)

    poses = a['poses']  # 维度 (1000, 165)
    trans = a['trans']  # 假设是平移信息

    rep = get_motion_representations(poses,trans)
    rec_poses, rec_trans = from_representations_to_motion(rep)

    save_dict = {}

    # Add all original items except poses and trans
    for key in a.keys():
        if key not in ['poses', 'trans']:
            save_dict[key] = a[key]

    # Add the modified poses and trans
    save_dict['poses'] = rec_poses
    save_dict['trans'] = rec_trans

    np.savez('tmp_test.npz',**save_dict)