import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils import functions
from utils.data import H36MDataset
from utils.data import H36MDataset_pair
import torch.optim as optim
import model_confidences
from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng

from matplotlib import pyplot as plt
from utils import bone_length

from torch import nn

from datetime import datetime
import os

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = SimpleNamespace()

config.learning_rate = 0.0001
config.BATCH_SIZE = 32
config.N_epochs = 100

# weights for the different losses
config.weight_rep = 1
config.weight_view = 1
config.weight_camera = 0.1

#data_folder = './data/'
data_folder = './detections_and_morphing_network/data/'

config.datafile = data_folder + 'detections.pickle'

def loss_weighted_rep_no_scale(p2d, p3d, confs):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d

    loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss

def reprojection(p3d):
    # the weighted reprojection loss as defined in Equation 5

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d

    return p3d_scaled

# loading the H36M dataset
#my_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[5, 6, 7, 8])
#my_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[5])
my_dataset = H36MDataset_pair(config.datafile, normalize_2d=True, subjects=[5, 6, 7, 8])

train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

# load the skeleton morphing model as defined in Section 4.2
# for another joint detector it needs to be retrained -> train_skeleton_morph.py
#model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
model_skel_morph = torch.load('./detections_and_morphing_network/models/model_skeleton_morph_S1_gh.pt')
model_skel_morph.eval()

# loading the lifting network
model = model_confidences.Lifter().cuda()

params = list(model.parameters())

optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

cam_names = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0']#, 'cam1', 'cam2', 'cam3']

today = datetime.now()

save_path = "./models/" + "model_save_" + today.strftime('%Y%m%d_%H%M%S')
os.mkdir(save_path)

for epoch in range(config.N_epochs):

    for i, s in enumerate(train_loader):

        # not the most elegant way to extract the dictionary
        #print(s.keys())
        for cam in all_cams:
            jk_after_array = []
            jk_array = []
            for k in s.keys():

                sample = s[k]

                #print(sample.keys())
                poses_2d = {key:sample[key] for key in all_cams}
                #print(sample)

                inp_poses = torch.zeros((poses_2d[cam].shape[0] * len(all_cams), 32)).cuda()
                inp_confidences = torch.zeros((poses_2d[cam].shape[0] * len(all_cams), 16)).cuda()

                #print(poses_2d['cam0'].shape, " SHAPE")
                #print(poses_2d['cam0'], " cam0 HIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        
                # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
                '''
                fig, ax1 = plt.subplots(1, 1)
                fig.suptitle(' LOSS')
                for i in range(0, 32, 2):
                    #print(poses_2d['cam0'][0][i].cpu().detach().numpy().item(), " HIIIII")
                    ax1.scatter(poses_2d['cam0'][0,:][i].cpu().detach().numpy().item(), poses_2d['cam0'][0,:][i+1].cpu().detach().numpy().item())

                plt.show()
                '''

                cnt = 0
                for b in range(poses_2d[cam].shape[0]):
                    for c_idx, cam in enumerate(poses_2d):
                        inp_poses[cnt] = poses_2d[cam][b]
                        inp_confidences[cnt] = sample['confidences'][cam_names[c_idx]][b]
                        cnt += 1

                '''
                lift
                '''
                # morph the poses using the skeleton morphing network
                #print(inp_poses.shape, " INP POSES !!!!")
        
                inp_poses = model_skel_morph(inp_poses)

                # predict 3d poses
                pred = model(inp_poses, inp_confidences)
                pred_poses = pred[0]
                pred_cam_angles = pred[1]
                #print(pred_poses.shape)

                jk_array.append(pred_poses)
                # angles are in axis angle notation
                # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                #print(pred_poses.shape, " PRED BEFORE")
                pred_poses, theta, root = functions.random_rotation(pred_poses.reshape(pred_poses.shape[0], 3, 16))

                pred_poses = pred_poses.reshape(pred_poses.shape[0], -1)
                #print(pred_poses.shape, " PRED AFTER")

                pred_rot = rodrigues(pred_cam_angles)

                # reproject to original cameras after applying rotation to the canonical poses
                rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)

                #print(rot_poses.shape, " ROT !!!!!")
                #print(rot_poses)
                '''
                lift again
                '''
                inp_poses = model_skel_morph(reprojection(rot_poses))

                # predict 3d poses
                pred = model(inp_poses, inp_confidences)
                pred_poses = pred[0]
                pred_cam_angles = pred[1]

                # angles are in axis angle notation
                # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                pred_rot = rodrigues(pred_cam_angles)

                # reproject to original cameras after applying rotation to the canonical poses
                rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)

                #print(rot_poses.shape, "reverse before")
                rot_poses = functions.reverse_rotation(rot_poses.reshape(pred_poses.shape[0], 3, 16), theta, root).reshape(pred_poses.shape[0], -1)

                jk_after_array.append(rot_poses)
                # reprojection loss
                losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)

                # view-consistency and camera-consistency
                # to compute the different losses we need to do some reshaping
                pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48))
                pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)
                confidences_rs = inp_confidences.reshape(-1, len(all_cams), 16)
                inp_poses_rs = inp_poses.reshape(-1, len(all_cams), 32)
                rot_poses_rs = rot_poses.reshape(-1, len(all_cams), 48)

                # view and camera consistency are computed in the same loop
            #losses.view = 0
            #losses.camera = 0
            

            # get combined loss
            #print(jk_array[0].shape, " JK ARRAY ZERO !!")
            #print(jk_after_array[0].shape, " JK AFTER ZERO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            j_t_3d = bone_length.compute_bone_length(jk_after_array[0])
            j_t_k_3d = bone_length.compute_bone_length(jk_array[1])

            #print(j_t_3d.shape, " HIII")
            mse_fn = nn.MSELoss()

            losses.bone = mse_fn(j_t_3d, j_t_k_3d)
            losses.temp = torch.norm(functions.temporal_loss(jk_array[0], jk_array[1], jk_after_array[0], jk_after_array[1]))
            losses.loss = config.weight_rep * losses.rep

            #print(losses.bone, " HIII BONE")
            #print(losses.temp, " HIIIIII TEMP")
            
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.bone + \
                        config.weight_camera * losses.temp
            
            '''
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view + \
                        config.weight_camera * losses.camera
            '''
            optimizer.zero_grad()
            losses.loss.backward()

            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []

                #print(key, value.detach().cpu(), " VALUE!!!!!!!!!!")
                losses_mean.__dict__[key].append(value.detach().cpu())
            #print(losses_mean, " HEEEEEEEEEEAAAAAAAAAAAAAAAAAAAA")
            # print progress every 100 iterations

            if not i % 100:
                # print the losses to the console
                print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(i % 1000))

                # this line is important for logging!
                losses_mean = SimpleNamespace()
        

    # save the new trained model every epoch
    print("MODEL SAVED")
    #torch.save(model, './models/model_lifter.pt')
    torch.save(model, save_path + '/' + 'model_lifter_' + str(epoch) + '_.pt')

    scheduler.step()

print('done')
