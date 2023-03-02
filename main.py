import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import cv2





# read minst model from local file model_Mnist.pth
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model= Net()
model.load_state_dict(torch.load('model_Mnist.pth'))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# read image from camera return the image



# split the number from the image

def split_number(image):
    # set image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # set image to binary
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # find the contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # get the number
    number = []
    for i in range(0, len(contours)):
        # cut the number into square shape
        x, y, w, h = cv2.boundingRect(contours[i])
        if w<300 and h<300 and ((w > 30 and h > 10) or (w > 10 and h > 30)):
            y=max(0,y)
            x=max(x,0)
            number.append([x, y, w, h])
    return number

# get the number from the image
def get_number(image, x,y,w,h):
    ans = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
    # set image to binaryt
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #reverse the color
    binary = 255 - binary
    return binary

def tran_image_to_square(image):
    # tran the image to square using blank to fill
    height, width = image.shape
    max_size=int(max(height,width)*1.5)
    result = np.zeros((max_size,max_size), dtype=np.uint8)
    startpos_x = int((max_size - width) / 2)
    startpos_y = int((max_size - height) / 2)
    result[startpos_y:startpos_y+height, startpos_x:startpos_x+width] = image
    return result


# resize the image to 28*28


def resize_image(image):
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    # set image to gray
    return image

# predict the number from the image

def predict_number(binary):
   
    # reverse the color
    # set image to float
    binary = binary.astype(np.float32)
    # set image to tensor
    binary = torch.from_numpy(binary)
    # set image to 1*1*28*28
    binary = binary.view(1, 1, 28, 28)
    # set image to Variable
    binary = Variable(binary)
    # predict the number
    output = model(binary)
    # get the number
    number = output.data.max(1, keepdim=True)[1]
    return number

# draw the number and the frame on the image

def draw_number(image, number, x, y, w, h,replaceimg):
    # draw the frame
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # draw the number
    cv2.putText(image, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    #replace the image 
    # image[y:y+h, x:x+w]=replaceimg
    return image

# main function
cap = cv2.VideoCapture(0)
while True:
    # take a picture
    ret, image = cap.read()
    ans=image.copy()
    # split the number
    number = split_number(image)
    # predict the number
    cnt=10
    for i in number:
        x, y, w, h = i
        # get the number
        num = get_number(image, x, y, w, h)
        tmp=num.copy()
        # tran the image to square using blank to fill
        num = tran_image_to_square(num)
        # resize the number
        num = resize_image(num)
        # predict the number
        num = predict_number(num)
        # get the number
        num = num.numpy()
        num = num[0][0]
        # draw the number and the frame on the image
        #trans tmp to 3 channel
        tmp=cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
        ans = draw_number(ans, num, x, y, w, h,tmp)
    # show the image
    cv2.imshow('image', ans)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
