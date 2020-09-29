import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import model
from PIL import Image

net = torch.load("model.pt", map_location=torch.device('cpu'))
print(net)

a=np.asarray(net.block1_conv1.weight.detach().squeeze()*255.0)
print(a.shape)
im = Image.fromarray(a)
im.save('map/b1c1.png')
im = Image.fromarray(np.asarray(net.block1_conv2.weight.detach().squeeze()*255.0))
im.save('map/b1c2.png')
im = Image.fromarray(np.asarray(net.block2_conv1.weight.detach().squeeze()*255.0))
im.save('map/b2c1.png')
im = Image.fromarray(np.asarray(net.block2_conv2.weight.detach().squeeze()*255.0))
im.save('map/b2c2.png')
im = Image.fromarray(np.asarray(net.block3_conv1.weight.detach().squeeze()*255.0))
im.save('map/b3c1.png')
im = Image.fromarray(np.asarray(net.block3_conv2.weight.detach().squeeze()*255.0))
im.save('map/b3c2.png')
im = Image.fromarray(np.asarray(net.block3_conv3.weight.detach().squeeze()*255.0))
im.save('map/b3c3.png')
im = Image.fromarray(np.asarray(net.block4_conv1.weight.detach().squeeze()*255.0))
im.save('map/b4c1.png')
im = Image.fromarray(np.asarray(net.block4_conv2.weight.detach().squeeze()*255.0))
im.save('map/b4c2.png')
im = Image.fromarray(np.asarray(net.block4_conv3.weight.detach().squeeze()*255.0))
im.save('map/b4c3.png')
im = Image.fromarray(np.asarray(net.block5_conv1.weight.detach().squeeze()*255.0))
im.save('map/b5c1.png')
im = Image.fromarray(np.asarray(net.block5_conv2.weight.detach().squeeze()*255.0))
im.save('map/b5c2.png')
im = Image.fromarray(np.asarray(net.block5_conv3.weight.detach().squeeze()*255.0))
im.save('map/b5c3.png')
im = Image.fromarray(np.asarray(net.fc1.weight.detach().squeeze()*255.0))
im.save('map/fc1.png')
im = Image.fromarray(np.asarray(net.fc2.weight.detach().squeeze()*255.0))
im.save('map/fc2.png')
im = Image.fromarray(np.asarray(net.fc3.weight.detach().squeeze()*255.0))
im.save('map/fc3.png')

