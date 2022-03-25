import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from matplotlib import pyplot as plt

#0 left hip
#1 left knee
#2 left foot
#3 right hip
#4 right knee
#5 right foot
#6 middle hip
#7 neck
#8 nose
#9 head
#10 left shoulder
#11 left elbow
#12 left wrist
#13 right shoulder
#14 right elbow
#15 right wrist

class H36MDataset(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, fname, normalize_2d=True, subjects=[1, 5, 6, 7, 8]):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        # select subjects
        selection_array = np.zeros(len(self.data['subjects']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(selection_array, (np.array(self.data['subjects']) == s))

        self.data['subjects'] = list(np.array(self.data['subjects'])[selection_array])
        cams = ['54138969', '55011271', '58860488', '60457274']
        for cam in cams:
            self.data['poses_2d_pred'][cam] = self.data['poses_2d_pred'][cam][selection_array]
            self.data['confidences'][cam] = self.data['confidences'][cam][selection_array]
            if normalize_2d:
                print(self.data['poses_2d_pred'][cam].shape)
                print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16).shape)
                print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0], " POSE")
                self.data['poses_2d_pred'][cam] = (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16) -
                                              self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[:, :, [6]]).reshape(-1, 32)
                ###
                '''
                fig, ax1 = plt.subplots(1, 1)
                fig.suptitle(' LOSS')
                for i in range(0, 16):
                    #print(poses_2d['cam0'][0][i].cpu().detach().numpy().item(), " HIIIII")
                    print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0].shape)
                    ax1.scatter(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i])
                    ax1.annotate(str(i), (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i]))
                plt.show()
                '''
                ###
                self.data['poses_2d_pred'][cam] /= np.linalg.norm(self.data['poses_2d_pred'][cam], ord=2, axis=1, keepdims=True)
                print(self.data['poses_2d_pred'][cam].shape)
                '''
                for i in range(0, 16):
                    #print(poses_2d['cam0'][0][i].cpu().detach().numpy().item(), " HIIIII")
                    print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0].shape)
                    ax1.scatter(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i])
                    ax1.annotate(str(i), (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i]))
                plt.show()
                stop
                '''
    def __len__(self):
        return self.data['poses_2d_pred']['54138969'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['54138969', '55011271', '58860488', '60457274']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(self.data['poses_2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d

        sample['confidences'] = dict()
        for cam in cams:
            sample['confidences'][cam] = torch.Tensor(self.data['confidences'][cam][idx].astype('float32')).cuda()

        sample['subjects'] = self.data['subjects'][idx]

        return sample

#NOT IMPLEMENTED YET
class dcpose(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, fname, normalize_2d=True, subjects=[1, 5, 6, 7, 8]):
        pickle_off = open(fname, "rb")
        self.data = pickle.load(pickle_off)

        # select subjects
        selection_array = np.zeros(len(self.data['subjects']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(selection_array, (np.array(self.data['subjects']) == s))

        self.data['subjects'] = list(np.array(self.data['subjects'])[selection_array])
        cams = ['54138969', '55011271', '58860488', '60457274']
        for cam in cams:
            self.data['poses_2d_pred'][cam] = self.data['poses_2d_pred'][cam][selection_array]
            self.data['confidences'][cam] = self.data['confidences'][cam][selection_array]
            if normalize_2d:
                print(self.data['poses_2d_pred'][cam].shape)
                print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16).shape)
                print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0], " POSE")
                self.data['poses_2d_pred'][cam] = (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16) -
                                              self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[:, :, [6]]).reshape(-1, 32)
                ###
                '''
                fig, ax1 = plt.subplots(1, 1)
                fig.suptitle(' LOSS')
                for i in range(0, 16):
                    #print(poses_2d['cam0'][0][i].cpu().detach().numpy().item(), " HIIIII")
                    print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0].shape)
                    ax1.scatter(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i])
                    ax1.annotate(str(i), (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i]))
                plt.show()
                '''
                ###
                self.data['poses_2d_pred'][cam] /= np.linalg.norm(self.data['poses_2d_pred'][cam], ord=2, axis=1, keepdims=True)
                print(self.data['poses_2d_pred'][cam].shape)
                '''
                for i in range(0, 16):
                    #print(poses_2d['cam0'][0][i].cpu().detach().numpy().item(), " HIIIII")
                    print(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0].shape)
                    ax1.scatter(self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i])
                    ax1.annotate(str(i), (self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][0][i], self.data['poses_2d_pred'][cam].reshape(-1, 2, 16)[0][1][i]))
                plt.show()
                stop
                '''
    def __len__(self):
        return self.data['poses_2d_pred']['54138969'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        cams = ['54138969', '55011271', '58860488', '60457274']

        for c_idx, cam in enumerate(cams):
            p2d = torch.Tensor(self.data['poses_2d_pred'][cam][idx].astype('float32')).cuda()
            sample['cam' + str(c_idx)] = p2d

        sample['confidences'] = dict()
        for cam in cams:
            sample['confidences'][cam] = torch.Tensor(self.data['confidences'][cam][idx].astype('float32')).cuda()

        sample['subjects'] = self.data['subjects'][idx]

        return sample
