import numpy as np

SMOOTH = 1e-6
from sklearn.metrics import jaccard_score

def metrics(output, target,input_id2):

    dice_result = []
    iou_result = []
    output = output[:,:,:,0]
    target = target[:, :, :, 0]

    for i in range(output.shape[0]):

        #if 'Normal' not in input_id2[i]:

        k = 1
        dice = round(np.sum(output[i][target[i] == k]) * 2.0 / (np.sum(output[i]) + np.sum(target[i])+1e-4),3)
        # dice_result += [dice]

        output_i = output[i].reshape(output.shape[1] * output.shape[2])
        target_i = target[i].reshape(target.shape[1] * target.shape[2])
        iou = round(jaccard_score(target_i, output_i, labels=np.unique(target_i), pos_label=1, average='binary'), 3)

        iou_result += [iou]
        dice_result += [dice ]

    return dice_result, iou_result


def metrics_eval(output, target, size1, size2):

    k = 1
    
    dice = round(np.sum(output[target == k]) * 2.0 / (np.sum(output) + np.sum(target)+1e-4),3)
            # dice_result += [dice]
    output_i = output.reshape(size1 * size2)
    target_i = target.reshape(size1 * size2)
    iou = round(jaccard_score(target_i, output_i, labels=np.unique(target_i), pos_label=1, average='binary'), 3)


    return dice, iou