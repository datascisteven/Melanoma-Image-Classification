import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
ImageFile.LOAD_TRUNCATED_IMAGES = True



def sixplot(var, auc, v_auc, pr_auc, v_pr_auc, prec, v_prec, rec, v_rec):
    """
        Function to produce the plots for accuracy, ROC-AUC, PR-AUC, recall, precision, and loss scores for the training and validation sets
        `var` is the variable that we set `model.fit` or `model.fit_generator`
        `auc`, `pr_auc`, `rec`, `prec`, etc. must be strings with quotes, i.e. `'auc_4'` or `'precision_8'`
    """

    f, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].plot(var.history['accuracy'], label = 'train')
    axs[0, 0].plot(var.history['val_accuracy'], label = 'validation')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()
    axs[0, 0].set_title('Accuracy Scores')
    
    axs[0, 1].plot(var.history[auc], label = 'train')
    axs[0, 1].plot(var.history[v_auc], label = 'validation')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('ROC-AUC')
    axs[0, 1].legend()
    axs[0, 1].set_title('ROC/AUC Scores')
    
    axs[0, 2].plot(var.history[pr_auc], label = 'train')
    axs[0, 2].plot(var.history[v_pr_auc], label = 'validation')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('PR-AUC')
    axs[0, 2].legend()
    axs[0, 2].set_title('PR/AUC Scores')
    
    axs[1, 0].plot(var.history[prec], label = 'train')
    axs[1, 0].plot(var.history[v_prec], label = 'validation')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].set_title('Precision Scores')
    
    axs[1, 1].plot(var.history[rec], label = 'train')
    axs[1, 1].plot(var.history[v_rec], label = 'validation')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()
    axs[1, 1].set_title('Recall Scores')
    
    axs[1, 2].plot(var.history['loss'], label = 'train')
    axs[1, 2].plot(var.history['val_loss'], label = 'validation')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].legend()
    axs[1, 2].set_title('Loss Scores')

    f.tight_layout(h_pad=5, w_pad=5)



def make_confusion_matrix(y, y_pred):
    cnf = confusion_matrix(y, y_pred)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})

def train_val_metrics(epochs, model_training):
    epochs = range(1, epochs+1)
    metrics = model_training.history
    train_loss = metrics['loss']
    train_acc = metrics['acc']
    val_loss = metrics['val_loss']
    val_acc = metrics['val_acc']
    
    ax = plt.subplot(211)
    train, = ax.plot(epochs, train_loss)
    val, = ax.plot(epochs, val_loss)
    ax.legend([train, val], ['training', 'validation'])
    ax.set(xlabel='epochs', ylabel='categorical cross-entropy loss')

    ax2 = plt.subplot(212)
    train2, = ax2.plot(epochs, train_acc)
    val2, = ax2.plot(epochs, val_acc)
    ax2.legend([train2, val2], ['training', 'validation'])
    ax2.set(xlabel='epochs', ylabel='accuracy')