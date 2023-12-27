
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import interp
from itertools import cycle


class AnalyzeROC():
    def __init__(self) -> None:
        ...
    
    def ROC_binary(self, 
                   y_test: np.array, 
                   y_pred: np.array, 
                   exp_path: str
    ):
        '''
        
        Args:
            ...
            
        Return:
            ...
        '''
        
        # 
        
        fp,tp,_ = roc_curve(y_test, y_pred)
        auc_score = auc(fp, tp)        
        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["font.size"] = 10
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fp, tp, label='Model (AUC = {:.4f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join('bin', 'exp', exp_path, 'ROC_curve.jpg'), dpi=300) 
        # plt.show()
        
        # Zoom in view of the upper left corner.
        plt.figure()
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot(fp, tp, label='Model (AUC = {:.4f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        plt.savefig(os.path.join('bin', 'exp', exp_path, 'ROC_curve_Zoom.jpg'), dpi=300) 
        # plt.show()
        
    def ROC_multi(self, 
                  y_true: np.array, 
                  y_pred: np.array,
                  exp_path: str, 
                  class_box: list
    ):
        '''
        
        Args:
            ...
            
        Return:
            ...
        '''
        
        # Number of classes
        n_classes = len(class_box)    

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        auc_score = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            auc_score[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', 
                linestyle='solid', 
                linewidth=4,
        )

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', 
                linestyle='dotted', 
                linewidth=4,
        )
        colors = cycle(['bisque', 
                        'tan', 
                        'rosybrown', 
                        'lightgreen', 
                        'dodgerblue', 
                        'royalblue', 
                        'purple', 
                        'pink', 
                        'cyan', 
                        'yellow'
                        ]
        )
        lss = cycle([(0,(1,0.5)),
                    (0,(0.5,1)),
                    (0,(1.5,1)),
                    (0,(1,1.5)),
                    (0,(2,1.5)),
                    (0,(1.5,2)),
                    ]
        )
        for i, color, ls in zip(range(n_classes), colors, lss):
            plt.plot(fpr[i], tpr[i],
                    label='{0} (area = {1:0.2f})'
                    ''.format(class_box[i], roc_auc[i]),
                    color=color, 
                    linestyle=ls,
                    linewidth=2,
            )

        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join('bin', 'exp', exp_path, 'roc_curve.jpg'),
                    dpi=300,
                    bbox_inches='tight',
        )   
        
        plt.close()
        
        return auc_score
        
    def ROC_multi_all(self, 
                      n_classes: list, 
                      y_true: np.array, 
                      y_pred: np.array, 
                      name_lst: list,
    ):
        '''
        
        Args:
            ...
            
        Return:
            ...
        '''
        
        # Plot linewidth.
        lw = 2
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        auc_score = dict()
        n_classes = len(n_classes)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            auc_score[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro (AUC = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                label='macro (AUC = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)
        colors = cycle(['bisque', 'tan', 'rosybrown', 
                        'lightgreen', 'dodgerblue', 'royalblue', 
                        'purple', 'pink', 'cyan', 'yellow'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='{0} (AUC = {1:0.2f})'
                    ''.format(name_lst[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('bin', 'exp', 'sample', 'roc_curve.jpg'))   
        plt.close()
        
        return auc_score
    
    def ROC_multi_ci(self, 
                     class_box: list, 
                     y_true: list, 
                     y_pred: list,
        ):
        '''
        
        Args:
            ...
            
        Return:
        '''
        
        n_classes = len(class_box)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        auc_score = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            auc_score[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
        return roc_auc["macro"]
    
    def ROC_multi_video(self, class_dict: dict, y_true: list, y_pred: list, now_time: str, plot_name: str):
        '''
        
        Args:
            ...
            
        Return:
            ...
        '''
        
        # Plot linewidth.
        lw = 2

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_dict)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_dict))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_dict)):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(class_dict)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 
                        'cornflowerblue', 'limegreen', 
                        'violet', 'yellow', 
                        'chocolate', 'olive'])
        for i, color in zip(range(len(class_dict)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(list(class_dict.keys())[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join('bin', 'exp_v', now_time, f'{plot_name}_roc_curve.jpg'))   
        
        plt.close()
        
        # Zoom in view of the upper left corner.
        # plt.figure(2)
        # plt.xlim(0, 0.2)
        # plt.ylim(0.8, 1)
        # plt.plot(fpr["micro"], tpr["micro"],
        #         label='micro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["micro"]),
        #         color='deeppink', linestyle=':', linewidth=4)

        # plt.plot(fpr["macro"], tpr["macro"],
        #         label='macro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["macro"]),
        #         color='navy', linestyle=':', linewidth=4)

        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #             label='ROC curve of class {0} (area = {1:0.2f})'
        #             ''.format(i, roc_auc[i]))

        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc="lower right")
        # plt.show()

    def CI_calc(self, y_true: list, y_pred: list):
        n_bootstraps = 10000
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for m in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.array(sorted(bootstrapped_scores))

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        auc = roc_auc_score(y_true, y_pred)
        aucCI_lower = sorted_scores[int(0.05 * len(bootstrapped_scores))]
        aucCI_upper = sorted_scores[int(0.95 * len(bootstrapped_scores))]
    
        return auc, aucCI_lower, aucCI_upper
