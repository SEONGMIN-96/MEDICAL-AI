import torch
import glob
import shutil

print ('Number of available devices ', torch.cuda.device_count())

# GPU 할당 변경하기
gpu_ids = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{gpu_ids}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Current devices ', torch.cuda.current_device())

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
from loss_function import *
from dataload import *
from utils import *
from metrics import *
import pandas as pd
# from model import *

# I expect to see RuntimeWarnings in this block

## 트레이닝 파라메터 설정하기
lr = 1e-3
batch_size = 2
num_epoch = 300

print(os.getcwd())
# os.chdir('/home/ubuntu/gcubme4/Workspace/S_Oh')
data_dir =  './data/220320_Dataset'
ckpt_dir = './results/20220320_segmentation/checkpoint'

log_dir = './results/20220320_segmentation/log'
result_dir = './results/20220320_segmentation/result'
save_dir = './results/20220320_segmentation/result_img'
mode = 'train'
train_continue = 'off'
kf_type = True

#device = 'cpu'
print("device: ",device)
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

    img = img/255

    out1 = medfilt(img, 3)

    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]).astype('uint8')

    out2 = scipy.ndimage.morphology.binary_fill_holes(out1, structure=kernel)
    out2 = out2*255

    return out2


def makeMask(img, w, h):

    maskpoint = np.array([[xx, yy] for xx, yy in zip(w, h)], dtype=np.int32)
    mask = cv2.fillPoly(img, [maskpoint], (255, 255, 255))

    return mask

from sklearn.metrics import jaccard_score

def IoU_cal(output,target):

    output_i = (output.reshape(output.shape[1] * output.shape[0])/255)
    target_i = (target.reshape(target.shape[1] * target.shape[0])/255)
    iou = round(jaccard_score(target_i, output_i, labels=np.unique(target_i), pos_label=1, average='binary'), 3)

    return iou
from torchvision import transforms, datasets, models

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
# early_stopping object의 초기화

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

    cv_num = [0,1,2,3,4]
    train = []

    for i in range(0,5):
        num = fold+i
        if num >=5:
            num = num-5
        if i <= 2:
            train_i = os.path.join(data_dir, 'CV'+str(cv_num[num]+1))
            train += [train_i]
        if i ==3:
            val = [os.path.join(data_dir, 'CV'+str(cv_num[num]+1))]
        if i == 4:
            test =  [os.path.join(data_dir, 'CV'+str(cv_num[num]+1))]

    return train, val, test



if mode == 'train':
     
    valid_loss_list = []
    train_loss_list = []
    kfold = KFold(n_splits=k_folds, shuffle=False)
    transform = transforms.Compose([transforms.RandomAffine(degrees=(0, 0), translate=(0.125, 0.125), scale=(0.75, 1.25)),
                                    transforms.RandomRotation(12.5, expand=False),transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
                                    Normalization(), ToTensor()])
    transform_val = transforms.Compose([Normalization(), ToTensor()])

    print('-------------')
    #print(dir(dataset_train))
    print('-------------')
    loss_result_list = pd.DataFrame()

    for fold in range(4,5):

        FOLD = fold + 1
        trainPath, valPath, testPath = get_cv(fold, data_dir)

        dataset_train = Dataset(data_dir=trainPath, transform=transform, target_transform=transform)
        dataset_val = Dataset(data_dir=valPath, transform=transform_val , target_transform=transform_val)

        best_epoch = 0

        deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        deeplab.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        deeplab.classifier = DeepLabHead(2048, 1)
        net = deeplab.to(device)
        ## 손실함수 정의하기
        fn_loss =BCEDiceLoss().to(device)
        ## Optimizer 설정하기
        ## Optimizer 설정하기
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,  weight_decay=0.0001)
        ## Ealry stopping
        ES_patience = 20
        early_stopping = EarlyStopping_CV(patience=ES_patience, verbose=True)
        valid_loss_list = []
        ## Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5,min_lr=1e-6,
                                                               patience=8, verbose=True)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  num_workers=0)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,  num_workers=0)
        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)

        if train_continue == "on":
            net, optim= load(ckpt_dir=ckpt_dir, net=net, optim=optim, fold = FOLD , KF= KF_TYPE)

        for epoch in range(st_epoch + 1, num_epoch + 1):

            count = 0
            net.train()
            loss_arr = []
            for batch, data in enumerate(loader_train, 1):
                count +=1
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                
                if input.size(dim=0) == 1:
                    input = torch.cat((input, input),0)
                    label = torch.cat((label, label),0)

                output = net(input)['out']
                sig = nn.Sigmoid()
                output = sig(output)
                input_id = data['Input_ID']

                # backward pass
                optim.zero_grad()

                loss = fn_loss(output , label)
                loss.backward()
                optim.step()
                # 손실함수 계산
                loss_arr += [loss.item()]

                print("TRAIN: K-Fold %d | EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (FOLD, epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(input)
                output = fn_tonumpy(fn_class(output))


            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

            with torch.no_grad():
                net.eval()
                loss_arr = []
                valid_loss = []
                dice_arr = []
                iou_arr = []
                results = []
                id = 0
                count = 0

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    if input.size(dim=0) == 1:
                        input = torch.cat((input, input),0)
                        label = torch.cat((label, label),0)

                    input_id = data['Input_ID']
                    input_id2 = [i.split('_')[-1].split('.')[0] for i in input_id]
                    output = net(input)['out']
                    sig = nn.Sigmoid()
                    output = sig(output)
                    loss = fn_loss(output, label)

                    # 손실함수 계산하기

                    # Tensorboard 저장하기
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(input)
                    output = fn_tonumpy(fn_class(output))
                    Dice, IoU = metrics(output, label,input_id)
                    loss_arr += [loss.item()]
                    dice_arr += Dice
                    iou_arr += IoU

                    print("VALID:Fold %d EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (FOLD, epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            scheduler.step(np.mean(loss_arr))
            print('Epoch-{0} lr: {1}'.format(epoch, optim.param_groups[0]['lr']))
            valid_loss = np.mean(loss_arr)
            valid_loss_list += [valid_loss]

            results = pd.DataFrame(
                {'Dice': dice_arr, 'IoU': iou_arr})
            results.to_csv(os.path.join(result_dir ,'val', 'cv'+str(FOLD), 'epoch'+str(epoch)+'_result.csv'))

            print("Validation Loss :",valid_loss)
            print("Dice", np.mean(dice_arr))
            print("IoU", np.mean(iou_arr))
            writer_val.add_scalar('loss', valid_loss, epoch)

            early_stopping(ckpt_dir, os.path.join(result_dir), results, valid_loss, net,optim, FOLD , epoch, kf=KF_TYPE)
            if early_stopping.early_stop:
                print("best epoch")
                best_epoch = epoch - ES_patience
                break

            ### EACH FOLD AND EPOCH
            loss_result = []
            loss_result = pd.DataFrame(data={'Fold': FOLD, 'Epoch': epoch, 'loss': valid_loss, 'LR': optim.param_groups[0]['lr'],
                                   'best epoch':best_epoch},index =[0])
            loss_result_list = loss_result_list.append(loss_result)
            loss_result_list.to_csv(os.path.join(result_dir, 'val', 'cv'+str(FOLD), 'CV_loss.csv'))
            #save_best_npy(FOLD,testPath)

        writer_train.close()
        writer_val.close()