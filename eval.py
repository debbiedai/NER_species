from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np
import os

def evaluate(path_label, path_pred):
    pred_list, label_list = [], []
    all_data_length = len(os.listdir(path_label))
    for i in range(all_data_length):
        # prediction files
        with open(os.path.join(path_pred, str(i), 'test_predictions.txt'), 'r') as f:
            for line in f:
                if line == "" or line == "\n":
                    continue
                else:
                    pred_list.append(line.split(" ")[-1][0])
        # label files
        with open(os.path.join(path_label, str(i), 'test.txt'), 'r') as fl:
            for line in fl:
                if line == "" or line == "\n":
                    continue
                else:
                    label_list.append(line.split(" ")[-1][0])
    return {
    "precision": precision_score([label_list], [pred_list]),
    "recall": recall_score([label_list], [pred_list]),
    "f1": f1_score([label_list], [pred_list]),
    }

result = evaluate('./LOOCV_dataset', './all_pred')
print(result)
