import torch

def compute_bone_length(X):
    """
    computes bone lengths of the bones in the skeleton
    X is 3D joint estimates of shape N x H x K x C
    where N is the number of frames, H is the number of humans in the frame (1 in case of Human3.6M),
    K is the number of joints and C is x,y,z coordinates
    
    """
    X = X.reshape(X.shape[0], 3, 16)
    X = torch.unsqueeze(torch.transpose(X,1,2), 1)
    #shape N x H x B where B is the number of bones in the skeleton
    skel = torch.zeros((X.shape[0], X.shape[1], 16))
    joint_pairs_indices = [[0,1], [1,2], [3,4],[4,5], [3,6], [6,0], [13,3], [10, 0], [13, 7], [7, 10], [8, 7], [9, 8], [10, 11], [11,12],[13,14],[14,15]]
    count = 0
    for bone in joint_pairs_indices:
        skel[:, :, count] = (torch.sum(((X[:,:,bone[0]] - X[:,:,bone[1]])**2), dim = -1)**0.5)
        count += 1
    return skel         

    #loss calculation lambda_bone * criterion(skel[frame][human], skel[prev_frame][prev_human]).sum()