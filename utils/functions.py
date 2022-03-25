import torch
import math

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

def random_rotation(J3d):
  J = J3d # need copy????
  batch_size = J.shape[0]
  theta = torch.rand(batch_size).cuda() * 2*torch.tensor(math.pi).cuda() # random theta
  root = J[:,:,8] # joint 8 = nose is root
  J3d_R = rotation(J.cuda(), theta.cuda(), root.unsqueeze(-1).cuda(), False)
  return J3d_R, theta, root # need these values in the code

def rotation(J, theta, root, is_reversed): # rotation over y axis by theta
  D =  root[:,2].cuda() # absolute depth of the root joint
  batch_size = root.shape[0]
  v_t = torch.zeros((batch_size, 3, 1)).cuda()
  v_t[:, 2, :] = D.cuda() # translation vector
  if is_reversed:
    root, v_t = v_t, root # swap
    theta = -theta
  # R = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]) # rotation matrix over z by theta degrees
  R = torch.zeros((batch_size, 3, 3)).cuda() # rotation matrix over y by theta degrees
  R[:, 0, 0] = torch.cos(theta)
  R[:, 0, 2] = torch.sin(theta)
  R[:, 1, 1] = torch.ones(batch_size)
  R[:, 2, 0] = -torch.sin(theta)
  R[:, 2, 2] = torch.cos(theta)
  # R = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]]) # rotation matrix over y by theta degrees
  # R = torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]]) # rotation matrix over x by theta degrees
  
  J_R = torch.matmul(R, J - root) + v_t # rotation
  return J_R

def reverse_rotation(J3d_R, theta, root):
  J = J3d_R # need copy????
  return rotation(J.cuda(), theta.cuda(), root.unsqueeze(-1).cuda(), True)


'''
def random_rotation(J3d):
  # J = torch.transpose(J3d, 1, 2)
  J = J3d
  root = torch.zeros(J.shape[0:2])
  for i in range(J.shape[0]):
    theta = torch.rand(1).cuda() * 2*torch.tensor(math.pi).cuda() # random theta
    root[i] = J[i,:,8] # joint 8 = nose is root
    temp = rotation(J[i,:,:], theta, root[i].unsqueeze(1), False)
    # print(temp.shape)
    J[i,:,:] = temp
  return J, theta, root # need these values in the code

def rotation(J, theta, root, is_reversed): # rotation over y axis by theta
  D =  root[2] # absolute depth of the root joint
  v_t = torch.tensor([[0], [0], [D]]).cuda() # translation vector
  if is_reversed:
    root, v_t = v_t, root # swap
    theta = -theta
  # R = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]) # rotation matrix over z by theta degrees
  R = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]]).cuda() # rotation matrix over y by theta degrees
  # R = torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]]) # rotation matrix over x by theta degrees
  
  J_R = torch.matmul(R, J.cuda() - root.cuda()) + v_t # rotation
  return J_R

def reverse_rotation(J3d_R, theta, root):
  # J = torch.transpose(J3d_R, 1, 2)
  J = J3d_R
  for i in range(J.shape[0]):
    J[i,:,:] = rotation(J[i,:,:].cuda(), theta.cuda(), root[i].unsqueeze(1).cuda(), True)
  return J
'''