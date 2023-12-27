import torch
import glob
import shutil
import warnings

import pandas as pd
# from model import *
import sklearn

from efficientnet.model import EfficientNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
from loss_function import *
from dataload_classification import *
from utils_classification import *
from metrics import *

# GPU 할당 변경하기
gpu_ids = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{gpu_ids}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Current devices ', torch.cuda.current_device())

# I expect to see RuntimeWarnings in this block

## 트레이닝 파라메터 설정하기
lr = 1e-4
batch_size = 32
num_epoch = 300

# os.chdir('/home/ubuntu/gcubme_ai/Workspace/S_Oh/US_Liver')
print(os.getcwd())

data_dir = './data/20220320_classification'
ckpt_dir = './result/20220320_classification_wc_eff_b2/checkpoint'

log_dir = './result/20220320_classification_wc_eff_b2/log'
result_dir = './result/20220320_classification_wc_eff_b2/result'

mode = 'train'
train_continue = 'off'
kf_type = True

# device = 'cpu'
print("device: ", device)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)
print("GPU ID : %s" % gpu_ids)
print("K-fold cross validation : %s" % str(kf_type))

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))
    os.makedirs(os.path.join(result_dir, 'val'))
    
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir, 'train')
    os.makedirs(log_dir, 'val')

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
# early_stopping object의 초기화

# TEST MODE
# TEST MODE

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import scipy

def defineName(ID):
    number = ID.split('.')[0]

    if len(number) == 1:
        ID = '0000' + ID
    elif len(number) == 2:
        ID = '000' + ID
    elif len(number) == 3:
        ID = '00' + ID
    elif len(number) == 4:
        ID = '0' + ID

    return ID

from scipy.signal import medfilt

def postprocessing(img):
    print(img.shape)

    img = img / 255

    out1 = medfilt(img, 3)

    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]).astype('uint8')

    out2 = scipy.ndimage.morphology.binary_fill_holes(out1, structure=kernel)
    out2 = out2 * 255

    return out2

def makeMask(img, w, h):
    maskpoint = np.array([[xx, yy] for xx, yy in zip(w, h)], dtype=np.int32)
    mask = cv2.fillPoly(img, [maskpoint], (255, 255, 255))

    return mask

from sklearn.metrics import jaccard_score

def IoU_cal(output, target):
    output_i = (output.reshape(output.shape[1] * output.shape[0]) / 255)
    target_i = (target.reshape(target.shape[1] * target.shape[0]) / 255)
    iou = round(jaccard_score(target_i, output_i, labels=np.unique(target_i), pos_label=1, average='binary'), 3)

    return iou

from torchvision import transforms, datasets, models

## 네트워크 학습시키기
st_epoch = 0
deepsuper = False
# TRAIN MODE
from sklearn.model_selection import KFold

k_folds = 5
KF_TYPE = True

# TEST MODE
# TEST MODE

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import scipy

def get_cv(fold, data_dir):
    cv_num = [0, 1, 2, 3, 4]
    train = []

    for i in range(0, 5):
        num = fold + i
        if num >= 5:
            num = num - 5
        if i <= 2:
            train_i = os.path.join(data_dir, 'CV' + str(cv_num[num] + 1))
            train += [train_i]
        if i == 3:
            val = [os.path.join(data_dir, 'CV' + str(cv_num[num] + 1))]
        if i == 4:
            test = [os.path.join(data_dir, 'CV' + str(cv_num[num] + 1))]

    return train, val, test

if mode == 'train':

    classes = ('0','1','2')
    classes_name = ['0','1','2']

    valid_loss_list = []
    train_loss_list = []
    kfold = KFold(n_splits=k_folds, shuffle=False)
    transform = transforms.Compose([transforms.RandomAffine(degrees=(0, 0), translate=(0.25, 0.25), scale=(0.8, 1.20)),
                                    transforms.RandomRotation(25.0, expand=False),transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
                                    Normalization(), ToTensor()])
    #transform = transforms.Compose([Normalization(), ToTensor()])
    transform_val = transforms.Compose([Normalization(), ToTensor()])

    print('-------------')
    # print(dir(dataset_train))
    print('-------------')
    loss_result_list = pd.DataFrame()

    for fold in range(0, 5):

        FOLD = fold + 1
        trainPath, valPath, testPath = get_cv(fold, data_dir)

        dataset_train = Dataset(data_dir=trainPath, classes=classes, ex = 4,transform=transform, target_transform=transform,s=96)
        dataset_val = Dataset(data_dir=valPath, classes=classes, ex = 4,transform=transform_val,
                              target_transform=transform_val,s=96)

        best_epoch = 0

        L_1 = []
        L_2 = []
        L_3 = []
        for i in range(0, 3):
            L_1 += glob.glob(os.path.join(trainPath[i], '*class_0_img*')) 
            L_2 += glob.glob(os.path.join(trainPath[i], '*class_1_img*')) 
            L_3 += glob.glob(os.path.join(trainPath[i], '*class_2_img*')) 

        nSamples = [len(L_1),len(L_2),len(L_3)]
        print('nSamples: ', nSamples)
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights)
        fn_loss = nn.CrossEntropyLoss(weight= normedWeights).to(device)

        #resnet = models.resnet101(pretrained=True)
        #num_classes = 2
        #resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #num_ftrs = resnet.fc.in_features
        #resnet.fc = nn.Linear(num_ftrs, num_classes)
        #net = resnet.to(device)
        #net = ConvNeXt(in_chans=1, num_classes=3,depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]).to(device)
        #net = convnext_base(in_chans, num_classes,pretrained=True).to(device)
        num_classes = 3
        
        model  = EfficientNet.from_pretrained('efficientnet-b2',in_channels = 1,num_classes= num_classes)
        net = model.to(device)
        print(net)

        ## 손실함수 정의하기

        ## Optimizer 설정하기
        optim = torch.optim.Adam(net.parameters(), lr=lr )
        #optim = torch.optim.SGD(net.parameters(), lr=lr, momentum = 0.9, weight_decay = 0.0001)
        ## Ealry stopping
        ES_patience = 30
        early_stopping = EarlyStopping_CV(patience=ES_patience, verbose=True)
        valid_loss_list = []
        ## Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, min_lr=1e-6,
                                                               patience=20, verbose=True)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)

        #if train_continue == "on":
        #    net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim, fold=FOLD, KF=KF_TYPE)

        for epoch in range(st_epoch + 1, num_epoch + 1):

            count = 0
            net.train()

            loss_arr = []

            f1_arr = []
            recall_arr = []
            precision_arr = []

            for batch, data in enumerate(loader_train, 1):
                count += 1
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                if input.size(dim=0) == 1:
                    input = torch.cat((input, input), 0)
                    label = torch.cat((label, label), 0)

                output = net(input)
          
                input_id = data['Input_ID']
                # backward pass
                optim.zero_grad()
                loss = fn_loss(output, label)
                loss.backward()
                optim.step()

                y_pred_softmax = torch.softmax(output, dim=1)
   
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

                print(label)
                print(y_pred_tags)

                F1score = sklearn.metrics.f1_score(label.to('cpu').detach().numpy(),
                                                   y_pred_tags.to('cpu').detach().numpy(), average='macro')
                recall = sklearn.metrics.recall_score(label.to('cpu').detach().numpy(),
                                                      y_pred_tags.to('cpu').detach().numpy(), average='macro')
                precision = sklearn.metrics.precision_score(label.to('cpu').detach().numpy(),
                                                            y_pred_tags.to('cpu').detach().numpy(), average='macro')

                # 손실함수 계산
                loss_arr += [loss.item()]

                f1_arr += [F1score]
                recall_arr += [recall]
                precision_arr += [precision]

                print(
                    "TRAIN: K-Fold %d | EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f |F1 score %.4f| recall %.4f| precision %.4f" %
                    (fold + 1, epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr), 
                     np.mean(f1_arr), np.mean(recall_arr), np.mean(precision_arr)))

                # Tensorboard 저장하기
                label = label.to('cpu').detach().numpy()
                input = fn_tonumpy(input)
                output = output.to('cpu').detach().numpy()

            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
            net.eval()

            with torch.no_grad():
                

                loss_arr = []
                valid_loss = []
                results = []

                y_pred_list = []
                y_label_list = []
                f1_arr = []
                recall_arr = []
                precision_arr = []

                id = 0
                count = 0

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    if input.size(dim=0) == 1:
                        input = torch.cat((input, input), 0)
                        label = torch.cat((label, label), 0)

                    input_id = data['Input_ID']
                    input_id2 = [i.split('_')[-1].split('.')[0] for i in input_id]
                    output = net(input)

                    loss = fn_loss(output, label)

                    y_pred_softmax = torch.softmax(output, dim=1)
       
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    print(label)
                    print(y_pred_tags)
                    F1score = sklearn.metrics.f1_score(label.to('cpu').detach().numpy(),
                                                       y_pred_tags.to('cpu').detach().numpy(), average='macro')
                    recall = sklearn.metrics.recall_score(label.to('cpu').detach().numpy(),
                                                          y_pred_tags.to('cpu').detach().numpy(), average='macro')
                    precision = sklearn.metrics.precision_score(label.to('cpu').detach().numpy(),
                                                                y_pred_tags.to('cpu').detach().numpy(), average='macro')
                    label = label.to('cpu').detach().numpy()
                    input = fn_tonumpy(input)
                    output = output.to('cpu').detach().numpy()
                    loss_arr += [loss.item()]

                    f1_arr += [F1score]
                    recall_arr += [recall]
                    precision_arr += [precision]

                    y_pred_list += [y_pred_tags.to('cpu').detach().numpy()]
                    y_label_list += [label]

                    print(
                        "VALID:Fold %d EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | F1  score %.4f| recall %.4f| precision %.4f " %
                        (fold + 1, epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr), 
                         np.mean(f1_arr), np.mean(recall_arr), np.mean(precision_arr)))

            scheduler.step(np.mean(loss_arr))
            print('Epoch-{0} lr: {1}'.format(epoch, optim.param_groups[0]['lr']))
            valid_loss = np.mean(loss_arr)
            valid_loss_list += [valid_loss]

            writer_val.add_scalar('loss', valid_loss, epoch)

            early_stopping(ckpt_dir, os.path.join(result_dir), valid_loss, net, optim, FOLD)
            if early_stopping.early_stop:
                print("best epoch")
                best_epoch = epoch - ES_patience
                break
            ### EACH FOLD AND EPOCH

            if epoch ==1:
                loss_result = pd.DataFrame(
                    data={'Fold': FOLD, 'Epoch': epoch, 'loss': valid_loss, 'LR': optim.param_groups[0]['lr'],
                          'best epoch': best_epoch}, index=[0])
            else:
                loss_result_i = pd.DataFrame(
                    data={'Fold': FOLD, 'Epoch': epoch, 'loss': valid_loss, 'LR': optim.param_groups[0]['lr'],
                          'best epoch': best_epoch}, index=[epoch-1])
                loss_result =pd.concat([loss_result,loss_result_i ],axis= 0)

            loss_result.to_csv(os.path.join(result_dir, 'val', str(FOLD) + 'CV_loss.csv'))
            # save_best_npy(FOLD,testPath)

        writer_train.close()
        writer_val.close()