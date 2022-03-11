import argparse
from os.path import join as pjoin
import os
import glob
import cv2
import xlsxwriter
from sklearn.metrics import confusion_matrix
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=True,
	help="path to image results")
parser.add_argument("-n", "--sets", required=True,
	help="number of sets")
parsed_args = parser.parse_args()

NUM_SETS = int(parsed_args.sets)
RESULT_PATH = parsed_args.images

experiment_results = {}

for experiment in os.listdir(RESULT_PATH):
            
    num_images = 0
    
    precision_tot = 0
    recall_tot = 0
    accuracy_tot = 0

    for set_nr in range(NUM_SETS):
        
        current_experiment = pjoin(RESULT_PATH, experiment, "set{}".format(set_nr))
        print(current_experiment)
                
        for image in glob.glob(pjoin(current_experiment, "*.png")):
            num_images += 1
            
            images = cv2.imread(image, 0)
            h, w = images.shape
            
            width = int(w / 2)
            
            mask = images[:, :width]
            pred = images[:, width:]
            
            TN, FP, FN, TP = confusion_matrix(np.matrix.flatten(pred), np.matrix.flatten(mask)).ravel()
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            accuracy = (TP + TN) / (TP + FP + FN + TN)
            
            precision_tot += precision
            recall_tot += recall
            accuracy_tot += accuracy
            
    metrics = {
            'precision': precision_tot / num_images,
            'recall': recall_tot / num_images,
            'accuracy': accuracy_tot / num_images,
            'f1-score': 2 * precision_tot * recall_tot / (precision_tot + recall_tot) / num_images,
    }
    experiment_results[experiment] = metrics



columns = {
        'precision': 0,
        'recall': 1,
        'accuracy': 2,
        'f1-score': 3,
    }

for experiment_name, experiment in experiment_results.items():
    row = 1
    workbook = xlsxwriter.Workbook(pjoin(RESULT_PATH, "{}.xlsx".format(experiment_name)))
    worksheet = workbook.add_worksheet()
    for key, value in experiment.items():
        
        for metric, column in columns.items():
            worksheet.write(0, column, metric)
            
        # write operation perform
        worksheet.write(row, columns[key], value)

        # incrementing the value of row by one
        # with each iterations.
        #row += 1
            
    workbook.close()