import os
import random
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from PIL import ImageDraw
from torch.autograd import Variable
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize])
# TARGET: [isCat, isDog]
train_data_list = []
target_list = []
train_data = []
waited = False
files = listdir('train/')
for i in range(len(listdir('train/'))):
    if len(train_data) == 58 and not waited:
        waited = True
        continue
    f = random.choice(files)
    files.remove(f)
    img = Image.open("train/" + f)
    img_tensor = transforms(img)  # (3,256,256)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len(listdir('train/')) / 64))
        print('Percentage Done: ', 100 * len(train_data) / int(len(listdir('train/')) / 64), '%')
        if len(train_data) > 15:
            break


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 768)
        x = F.relu(self.dropout2(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)


model = Netz()

if os.path.isfile('meinNetz.pt'):
    model = torch.load('meinNetz.pt')

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


# optimizer = optim.Adam(model.parameters(), lr=0.01)
def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data),
                   100. * batch_id / len(train_data), loss.item()))
        batch_id = batch_id + 1


def test():
    model.eval()
    files = listdir('test_set/cats/')
    f = random.choice(files)
    img = Image.open('test_set/cats/' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor)
    out = model(data)
    # print(str(f) + ": " + str(out.data.max(1, keepdim=True)[1]))
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("Roboto-Bold.ttf", 50)
    # draw.text((x, y),"Sample Text",(r,g,b))
    text = "Cat"
    print(out.data.max(1, keepdim=True)[1].cpu().numpy()[0])
    if out.data.max(1, keepdim=True)[1].cpu().numpy()[0] != 0:
        text = "Dog"
    # draw.text((0, 0), text, (0, 0, 0))#, font=font)
    print('This is a ', text)
    img.show()
    letsgo = input('')


for epoch in range(1, 30):
    train(epoch)
    test()
    torch.save(model, 'meinNetz.pt')
