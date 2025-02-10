import random
import matplotlib.pyplot as plt
import numpy as np
import pdb
import math
import torch
import torch.nn.functional as F
from numba import jit

def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3, seed=0):
    """
    data_numpy: T,V,C
    """
    torch.manual_seed(seed)
    data_torch = torch.from_numpy(data_numpy)
    T, V, C = data_torch.shape
    data_torch = data_torch.permute(0,2,1).contiguous() # T,3,V
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.permute(0,2,1).contiguous()

    return data_torch

def split_data(data):
    # 定义子矩阵的形状
    submatrix_shape = (900, 20, 3)
    # 计算滑动步长
    step_size = (data.shape[0] - submatrix_shape[0]) // 15  # 总步数为16，包括初始位置
    new_data = np.zeros((16, 900, 20, 3))
    for i in range(new_data.shape[0]):
        new_data[i] = data[i * step_size:i * step_size + 900]
    
    return new_data

def split_data20s(data):
    # 定义子矩阵的形状
    submatrix_shape = (600, 20, 3)
    # 计算滑动步长
    step_size = (data.shape[0] - submatrix_shape[0]) // 15  # 总步数为16，包括初始位置
    new_data = np.zeros((16, 600, 20, 3))
    for i in range(new_data.shape[0]):
        new_data[i] = data[i * step_size:i * step_size + 600]
    
    return new_data

def split_data_8(data):
    if data.shape[0] < 1200:
        data = np.concatenate([data, data], axis=0)
    # 定义子矩阵的形状
    submatrix_shape = (1200, 20, 3)
    # 计算滑动步长
    step_size = (data.shape[0] - submatrix_shape[0]) // 7  # 总步数为8，包括初始位置
    new_data = np.zeros((8, 1200, 20, 3))
    for i in range(new_data.shape[0]):
        new_data[i] = data[i * step_size:i * step_size + 1200]
    
    return new_data

def padding_data(matrix):
    # 获取第一维的尺寸
    dim1_size = matrix.shape[0]

    # 如果第一维尺寸大于9000，则截断至9000
    if dim1_size > 9000:
        matrix = matrix[:9000, :, :]

    # 如果第一维尺寸小于9000，则填充至9000
    elif dim1_size < 9000:
        # 计算需要填充的数量
        padding_size = 9000 - dim1_size
        # 获取最后一个二维矩阵
        last_2d_matrix = matrix[-1, :, :]
        # 使用tile函数填充至9000
        padding = np.tile(last_2d_matrix, (padding_size, 1, 1))
        # 将填充后的矩阵与原矩阵合并
        matrix = np.concatenate((matrix, padding), axis=0)
    
    matrix = matrix.reshape((2,4500,20,3)) # [2, 4500, 20, 3]

    return matrix


def frame_translation(ske_joints,dimension="3D"):
    if dimension=="3D":
        num_frames = ske_joints.shape[0]
        for f in range(num_frames):
            origin = ske_joints[f, 17:18] #joint-17 is the root joint
            ske_joints[f] = ske_joints[f]-np.tile(origin, (ske_joints.shape[1], 1))
        return ske_joints
    elif dimension=="2D":
        poses = np.zeros((ske_joints.shape[0], 20, ske_joints.shape[2]))
        poses[:, 0:18] = ske_joints[:, 0:18]
        poses[:, 19] = (ske_joints[:, 11] + ske_joints[:, 12]) / 2
        poses[:, 18] = (poses[:, 17] + 2*poses[:, 19]) / 3
        num_frames = poses.shape[0]
        for f in range(num_frames):
            origin = poses[f, 17:18]
            poses[f] = poses[f]-np.tile(origin, (poses.shape[1], 1))
        
        return poses


def get_delta(x):
    df = np.concatenate((np.asarray([0]), np.diff(x))) * (np.asarray(x * 0) + 1)
    return df

def STAM_calculate_features(data):
    delta_t=0.033
    #计算关节点速度,加速度，位移
    velocity_x=np.zeros([data.shape[0],data.shape[1],1])
    velocity_y=np.zeros([data.shape[0],data.shape[1],1])
    displacement=np.zeros([data.shape[0],data.shape[1],1])
    acceleration_x=np.zeros([data.shape[0],data.shape[1],1])
    acceleration_y=np.zeros([data.shape[0],data.shape[1],1])
    for i in range(data.shape[1]):
        joint_x=data[:,i,0]
        joint_y=data[:,i,1]
        displacement_x=get_delta(joint_x)
        displacement_y=get_delta(joint_y)
        v_x= displacement_x / delta_t
        v_y= displacement_y / delta_t
        a_x= v_x / delta_t
        a_y= v_y / delta_t
        
        displacement[:,i,:]= np.sqrt(displacement_x ** 2 + displacement_y ** 2).reshape(data.shape[0],1)
        velocity_x[:,i,:]= v_x.reshape(data.shape[0],1)
        velocity_y[:,i,:]= v_y.reshape(data.shape[0],1)
        acceleration_x[:,i,:]= a_x.reshape(data.shape[0],1)
        acceleration_y[:,i,:]= a_y.reshape(data.shape[0],1) 
    # [T, V, 8]
    data_features=np.concatenate((data,velocity_x,velocity_y,acceleration_x,acceleration_y,displacement),axis=2) 

    if data_features.shape[0] < 1200:
        data_features = np.concatenate([data_features, data_features], axis=0)
    # 定义子矩阵的形状
    submatrix_shape = (1200, 20, data_features.shape[2])
    # 计算滑动步长
    step_size = (data_features.shape[0] - submatrix_shape[0]) // 7  # 总步数为8，包括初始位置
    new_data = np.zeros((8, 1200, 20, data_features.shape[2]))
    for i in range(new_data.shape[0]):
        new_data[i] = data_features[i * step_size:i * step_size + 1200]
    return new_data 


def cal_angle(point_a, point_b, point_c):
    """
    根据三点坐标计算夹角
    :param point_a: 点1坐标
    :param point_b: 点2坐标
    :param point_c: 点3坐标
    :return: 返回点2夹角值
    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]
    a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]
    x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
    x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

    cos_b = (x1*x2 + y1*y2 + z1*z2) / ((math.sqrt(x1**2 + y1**2 + z1**2)) *(math.sqrt(x2**2 + y2**2+z2**2)))
    if cos_b>1:
        cos_b=1
    if cos_b<-1:
        cos_b=-1
    B = math.degrees(math.acos(cos_b))
    return B


def angular_disp(x, y):
    possible_angles = np.asarray([y - x, y - x + 360, y - x - 360])
    idxMinAbsAngle = np.abs([y - x, y - x + 360, y - x - 360]).argmin(axis=0)
    smallest_angle = np.asarray([possible_angles[idxMinAbsAngle[i], i] for i in range(len(possible_angles[0]))])
    return smallest_angle


def calculate_features(data):
    # delta_t=0.033
    # #计算18个关节角
    # angle_colume=np.zeros([data.shape[0],data.shape[1],1])
    # for i in range(data.shape[0]):
    #     a=data[i,:,:]
    #     angle_colume[i,0,:]=cal_angle(a[2],a[0],a[17])
    #     angle_colume[i,1,:]=cal_angle(a[0],a[1],a[3])
    #     angle_colume[i,2,:]=cal_angle(a[0],a[2],a[4])
    #     angle_colume[i,3,:]=cal_angle(a[1],a[3],(1,0,0))
    #     angle_colume[i,4,:]=cal_angle(a[2],a[4],(1,0,0))
    #     angle_colume[i,5,:]=cal_angle(a[7],a[5],a[17])
    #     angle_colume[i,6,:]=cal_angle(a[8],a[6],a[17])
    #     angle_colume[i,7,:]=cal_angle(a[5],a[7],a[9])
    #     angle_colume[i,8,:]=cal_angle(a[6],a[8],a[10])
    #     angle_colume[i,9,:]=cal_angle(a[7],a[9],(1,0,0))
    #     angle_colume[i,10,:]=cal_angle(a[8],a[10],(1,0,0))
    #     angle_colume[i,11,:]=cal_angle(a[13],a[11],a[19])
    #     angle_colume[i,12,:]=cal_angle(a[14],a[12],a[19])
    #     angle_colume[i,13,:]=cal_angle(a[11],a[13],a[15])
    #     angle_colume[i,14,:]=cal_angle(a[12],a[14],a[16])
    #     angle_colume[i,15,:]=cal_angle(a[13],a[15],(1,0,0))
    #     angle_colume[i,16,:]=cal_angle(a[14],a[16],(1,0,0))
    #     angle_colume[i,17,:]=cal_angle(a[18],a[17],a[6])
    #     angle_colume[i,18,:]=cal_angle(a[17],a[18],a[19])
    #     angle_colume[i,19,:]=cal_angle(a[11],a[19],a[18])

    # #计算关节角速度
    # angle_velocity=np.zeros([data.shape[0],data.shape[1],1])
    # for i in range(angle_colume.shape[1]):
    #     angle=angle_colume[:,i,:]
    #     b = angular_disp(angle[0:len(angle) - 1], angle[1:len(angle)]).reshape(len(angle) - 1,1) / delta_t
    #     angle_velocity[:,i,:]=np.concatenate((np.zeros([1,1]), b))  

    # #计算关节点速度
    # velocity=np.zeros([data.shape[0],data.shape[1],1])
    # for i in range(data.shape[1]):
    #     joint_x=data[:,i,0]
    #     joint_y=data[:,i,1]
    #     joint_z=data[:,i,2]
    #     d_x= get_delta(joint_x) / delta_t
    #     d_y= get_delta(joint_y) / delta_t
    #     d_z= get_delta(joint_z) / delta_t
    #     velocity[:,i,:]= np.sqrt(d_x ** 2 + d_y ** 2 + d_z**2).reshape(data.shape[0],1)  

    #计算骨骼
    bone_pairs = (
    (0, 17), (1, 0), (2, 0), (3, 1), (4, 2), (5, 17),(6, 17), (7, 5), (8, 6), (9, 7), (10, 8), (11, 19),
    (12, 19), (13, 11), (14, 12), (15, 13), (16, 14), (17, 18),(18, 19)
    )
    data_bone=np.zeros(data.shape)
    for v1, v2 in bone_pairs:
        data_bone[:, v1] = data[:, v1]- data[:, v2]
    
    #合并
    # data_features=np.concatenate((data_bone,angle_colume,angle_velocity,velocity),axis=2)
    data_features=data_bone
    if data_features.shape[0] < 1200:
            data_features = np.concatenate([data_features, data_features], axis=0)
    # 定义子矩阵的形状
    submatrix_shape = (1200, 20, data_features.shape[2])
    # 计算滑动步长
    step_size = (data_features.shape[0] - submatrix_shape[0]) // 7  # 总步数为8，包括初始位置
    new_data = np.zeros((8, 1200, 20, data_features.shape[2]))
    for i in range(new_data.shape[0]):
        new_data[i] = data_features[i * step_size:i * step_size + 1200]
    
    return new_data 
    