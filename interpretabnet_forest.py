from pytorch_tabnet.tab_model import TabNetClassifier

#from tab_model import TabNetClassifier


import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)


import os
import wget
from pathlib import Path
import shutil
import gzip
import math

from matplotlib import pyplot as plt



def main():

    # Download ForestCoverType dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    dataset_name = 'forest-cover-type'
    dataset = 'forest-cover-type'
    tmp_out = Path('./data/'+dataset_name+'.gz')
    out = Path(os.getcwd()+'/data/'+dataset_name+'.csvpyth')
    
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, tmp_out.as_posix())
        with gzip.open(tmp_out, 'rb') as f_in:
            with open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Load data and split
    target = "Covertype"

    bool_columns = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]

    int_columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (
            int_columns + bool_columns + [target])

    # train = pd.read_csv('data/covtype.csv', header=None, names=feature_columns)
    train = pd.read_csv(out, header=None, names=feature_columns)
    # print("number of features")
    # print(len(feature_columns))

    n_total = len(train)

    # Train, val and test split follows
    # Rory Mitchell, Andrey Adinets, Thejaswi Rao, and Eibe Frank.
    # Xgboost: Scalable GPU accelerated learning. arXiv:1806.11248, 2018.

    train_val_indices, test_indices = train_test_split(
        range(n_total), test_size=0.2, random_state=0)
    train_indices, valid_indices = train_test_split(
        train_val_indices, test_size=0.2 / 0.6, random_state=0)

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns[train.dtypes == object]:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    for col in train.columns[train.dtypes == 'float64']:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    unused_feat = []

    features = [ col for col in train.columns if col not in unused_feat+[target]]

    # print(features)

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    # TUNING HYPERPARAMETERS 

    # for maximum 5 epoch, nd and na = 128 wins, nd and na = 64 doesn't have much
    # difference too
    nd_na = [16, 24, 32, 64, 128]
    n_steps = [4, 5, 6, 7, 8, 9, 10]
    gammas = [1.0, 1.2, 1.5, 2.0]

    # 0.001 is default
    lambda_sparses = [0, 0.000001, 0.0001, 0.01, 0.1]

    # batch default 1024
    # virtual batch default 128
    B = [256, 512, 1024, 2048, 4096, 16384, 32768]
    b_v = [256, 512, 1024, 2048, 4096]
    l_r = [0.005, 0.01, 0.02, 0.025]
    de_r = [0.4, 0.8, 0.9, 0.95]
    reg_ws = [0.001, 1, 10, 100, 1000]
    acc = []
#     for reg_w in reg_ws:
#         print("Tuning hyperparameters")
#         print("nd and na is 128, and nstep is 3, gamma is 1.5, and lambda is 0.001 , "
#             "and regularizer weight is:"
#             + str(reg_w))

#         clf = TabNetClassifier(
#             n_d=128,
#             n_a=128,
#             n_steps=3,
#             gamma=1.5,
#             lambda_sparse=0.001,
#             cat_idxs=cat_idxs,
#             cat_dims=cat_dims,
#         )

#         clf.fit(
#             X_train=X_train, y_train=y_train,
#             eval_set=[(X_train, y_train), (X_valid, y_valid)],
#             eval_name=['train', 'valid'],
#             max_epochs=2
#         )

#         y_pred = clf.predict(X_test)
#         test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
#         acc.append(test_acc)
#         print(f"FINAL TEST SCORE FOR forest cover type : {test_acc}")

#     print("Optimal regularizer weight is:" + str(reg_ws[acc.index(max(acc))]))

#     # if os.getenv("CI", False):
#     #     # Take only a subsample to run CI
#     #     X_train = train[features].values[train_indices][:1000,:]
#     #     y_train = train[target].values[train_indices][:1000]
#     # else:

#     max_epochs = 5 if not os.getenv("CI", False) else 2

#     # clf.fit(
#     #     X_train=X_train, y_train=y_train,
#     #     eval_set=[(X_train, y_train), (X_valid, y_valid)],
#     #     eval_name=['train', 'valid'],
#     #     max_epochs=max_epochs, patience=100,
#     #     batch_size=16384, virtual_batch_size=512
#     # )


#     # or you can simply use the predict method

#     explain_matrix, masks = clf.explain(X_test)

#     # print("masks")
#     # print(masks)

#     fig, axs = plt.subplots(1, 3, figsize=(20,20))

#     for i in range(3):
#         axs[i].imshow(masks[i][:50])
#         axs[i].set_title(f"mask {i}")
#         axs[i].set_xlabel("features")
#         axs[i].set_ylabel("50 samples")
#         axs[i].set_xticks(np.arange(0, 55, 5))

#     # axs[0].set_xticklabels(features)
#     plt.savefig('mask_visualization.png')

#     # investigate on incorrect predictions
#     incorr_pred_i = np.nonzero(y_pred - y_test)[0].flatten()
#     print("incorrect predictions")
#     print(incorr_pred_i)

#     figure, ax = plt.subplots(1, 3, figsize=(20,20))

#     # selecting incorrect predictions
#     for i in range(3):
#         masks[i] = pd.DataFrame(masks[i])
#         masks[i] = masks[i].loc[incorr_pred_i]
#         masks[i].to_numpy()

#     for i in range(3):
#         ax[i].imshow(masks[i][:50])
#         ax[i].set_title(f"mask {i}")
#         ax[i].set_xlabel("features")
#         ax[i].set_ylabel("Incorrect prediction samples")
#         ax[i].set_xticks(np.arange(0, 55, 5))
#     #ax[0].set_xticklabels(features)
#     plt.xticks(rotation='vertical')
#     plt.savefig("incorrect_prediction.png")

#     # # selecting correct predictions
#     # for i in range(3):
#     #     masks[i] = pd.DataFrame(masks[i])
#     #     masks[i] = masks[i].loc[-incorr_pred_i]
#     #     masks[i].to_numpy()

#     # for i in range(3):
#     #     ax[i].imshow(masks[i][:50])
#     #     ax[i].set_title(f"mask {i}")
#     #     ax[i].set_xlabel("features")
#     #     ax[i].set_ylabel("correct prediction samples")
#     #     ax[i].set_xticks(np.arange(0, 55, 5))
#     # #ax[0].set_xticklabels(features)
#     # plt.xticks(rotation='vertical')
#     # plt.savefig("correct_prediction.png")


#     cm = confusion_matrix(y_test, y_pred)

#     #print(cm)

#     # Show confusion matrix in a separate window
#     plt.matshow(cm)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig("confusion_matrix.png")

#     # for instance-wise example 1
#     # data = {'C':20, 'C++':15, 'Java':30,
#     #         'Python':35}
#     # courses = list(data.keys())
#     # values = list(data.values())

#     fig = plt.figure(figsize = (10, 8))

#     # creating the bar plot
#     plt.bar(features, explain_matrix[0], color ='maroon',
#             width = 0.4)

#     plt.xlabel("features")
#     plt.ylabel("feature importance")
#     plt.xticks(rotation='vertical')
#     if y_pred[0] == y_test[0]:
#         plt.title("Correctly predicts example" + str(y_pred[0]))
#     else:
#         plt.title("incorrectly predicts example" + str(y_pred[0]) + "actual" + str(y_test[0]))
#     plt.savefig("instance-wise feature selection0")



#     fig = plt.figure(figsize = (10, 8))

#     # creating the bar plot
#     plt.bar(features, explain_matrix[1], color ='maroon',
#             width = 0.4)

#     plt.xlabel("features")
#     plt.ylabel("feature importance")
#     plt.xticks(rotation='vertical')
#     if y_pred[1] == y_test[1]:
#         plt.title("Correctly predicts example" + str(y_pred[1]))
#     else:
#         plt.title("incorrectly predicts example" + str(y_pred[1]) + "actual" + str(y_test[1]))
#     plt.savefig("instance-wise feature selection1")


#     # print( "predictions")
#     # print(y_pred)
#     #
#     # print("true values")
#     # print(y_test)

#     # test sample 0, correctly predicted
#     fig, axs = plt.subplots(1, 3, figsize=(20,20))

#     for i in range(3):
#         axs[i].imshow(masks[i][:1])
#         axs[i].set_title(f"mask {i}")
#         axs[i].set_xlabel("features")
#         axs[i].set_ylabel("test sample #0")
#         axs[i].set_xticks(np.arange(0, 55, 5))

#     # axs[0].set_xticklabels(features)
#     plt.savefig("instance-wise_mask_example0.png")


#     # test sample 7, incorrectly predicted
#     fig, axs = plt.subplots(1, 3, figsize=(20,20))

#     for i in range(3):
#         axs[i].imshow(masks[i][7:8])
#         axs[i].set_title(f"mask {i}")
#         axs[i].set_xlabel("features")
#         axs[i].set_ylabel("test sample #7")
#         axs[i].set_xticks(np.arange(0, 55, 5))

#     # axs[0].set_xticklabels(features)
#     plt.savefig("instance-wise_mask_example7.png")

# Optimized Run #######################################################################################################################
    # Optimum Hyperparameters Training [128, 4, 1.5, 0.001, 0.02]
    n_steps = 4
    opt_ndna = 128
    opt_gamma = 1.5
    opt_lambda = 0.001
    opt_lr = 0.02
    opt_reg_m = 0

    def search_best_reg_m(start=0, end=1000000000, col_threshold_val=0.20, col_threshold=3, all_mask_pass=None, all_mask_pass_thresh=3, step_size=None, best_reg_m=None, reg_m_acc_dict=None, is_recursive=False):
        if reg_m_acc_dict is None:
                reg_m_acc_dict = {}

        if all_mask_pass == all_mask_pass_thresh:
                print(reg_m_acc_dict)
                final_reg_m = max(reg_m_acc_dict, key=reg_m_acc_dict.get)
                return final_reg_m
            
        if all_mask_pass is None:
                all_mask_pass = 0
            
        # Fine-tuning around the best found value
        best_reg_m = None
        break_outer_loop = False

        # Determining Magnitude for reg_m
        diff = end - start
        magnitude = int(math.log10(diff))

        reg_m = start
        while reg_m <= end and all_mask_pass < all_mask_pass_thresh: #do i need all_mask_pass threshold here?
                print("reg_m", reg_m)
                if reg_m in reg_m_acc_dict:
                    reg_m += step_size
                    continue
                clf = TabNetClassifier(
                    n_d=opt_ndna,
                    n_a=opt_ndna,
                    n_steps=4,
                    gamma=opt_gamma,
                    lambda_sparse=opt_lambda,
                    cat_idxs=cat_idxs,
                    cat_dims=cat_dims,
                    optimizer_params=dict(lr=opt_lr),
                    mask_type = 'softmax',
                    reg_m=reg_m
                )
                # max epoch 50
                clf.fit(
                    X_train=X_train, y_train=y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    eval_name=['train', 'valid'],
                    max_epochs=30, eval_metric=['accuracy']
                )

                y_pred = clf.predict(X_test)
                test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

                print(f"FINAL TEST SCORE FOR {dataset} : {test_acc}")

                explain_matrix, masks = clf.explain(X_test)

                # Extract the first 50 samples from each matrix
                masks_dict = {}
                for key, value in masks.items():
                    masks_dict[key] = value[:50]

                # Normalize each extracted matrix so that its sum is 1
                for key, value in masks_dict.items():
                    total_sum = value.sum()
                    
                    # Avoid division by zero
                    if total_sum == 0:
                        continue
                    
                    masks_dict[key] = value / total_sum

                mask_threshold = n_steps // 2 + 1
                mask_pass_count = 0

                for key, value in masks_dict.items():
                    column_sums = value.sum(axis=0)
                    # print(f"Sum of columns for matrix with key {key}: {column_sums}")

                    # Check which columns are greater than col_threshold_val
                    cols_above_threshold = [i for i, col_sum in enumerate(column_sums) if col_sum > col_threshold_val]
                    print(f"Columns in matrix with key {key} that are greater than the threshold value: {cols_above_threshold}")

                    if col_threshold-1 <= len(cols_above_threshold) <= col_threshold+1:
                        mask_pass_count += 1
                        print("Num Mask Pass Threshold:", mask_pass_count)
                    if mask_pass_count >= mask_threshold:
                        if len(reg_m_acc_dict) == 0:
                            all_mask_pass += 1
                            best_reg_m = reg_m
                            reg_m_acc_dict[reg_m] = test_acc
                            break_outer_loop = True
                            break
                        elif test_acc > max(reg_m_acc_dict.values()):
                            all_mask_pass += 1
                            reg_m_acc_dict[reg_m] = test_acc
                            best_reg_m = reg_m
                            break
                        else:
                            print("Lesser Acc, Break")
                            break
                    
                if break_outer_loop:
                    break

                if is_recursive:
                    reg_m += step_size
                elif reg_m == 0:
                    reg_m = 10
                else:
                    reg_m *= 10


        # Check conditions after looping over all possible reg_m values
        if best_reg_m is not None and len(reg_m_acc_dict) == 1: # i need to add the condition where i hit all mask pass and return the funct directly
                print('Breaked')
                magnitude = math.floor(math.log10(best_reg_m))
                if magnitude >= 1:
                    step_size = 10**(magnitude-1)
                else:
                    step_size = 10**(magnitude)
                # Recursively refine the search with updated boundaries and reduced depth
                new_start = int(max(start, best_reg_m - step_size))
                new_end = int(min(end, best_reg_m + step_size))
                return search_best_reg_m(new_start, new_end, col_threshold, col_threshold_val, all_mask_pass, all_mask_pass_thresh, step_size, best_reg_m, reg_m_acc_dict, is_recursive=True)
        elif len(reg_m_acc_dict)==0:
                return "Did not pass! Lower threshold!"
        else:
                final_reg_m = max(reg_m_acc_dict, key=reg_m_acc_dict.get)
                return final_reg_m
            
    opt_reg_m = search_best_reg_m()
    print("opt_reg_m for best mask", opt_reg_m)

    clf = TabNetClassifier(
        n_d=opt_ndna,
        n_a=opt_ndna,
        n_steps=4,
        gamma=opt_gamma,
        lambda_sparse=opt_lambda,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_params=dict(lr=opt_lr),
        mask_type = 'softmax',
        reg_m=opt_reg_m
    )
    # max epoch 50
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        max_epochs=100, eval_metric=['accuracy']
    )

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    print(f"FINAL TEST SCORE FOR {dataset} : {test_acc}")

    explain_matrix, masks = clf.explain(X_test)
    fig, axs = plt.subplots(1, n_steps, figsize=(20, 20))
    for i in range(n_steps):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
        axs[i].set_ylabel("Test Samples")
        axs[i].set_xlabel("Features")
    plt.savefig(f"{dataset}_feature_mask_kld_{opt_reg_m}_accuracy_{test_acc}.png")

if __name__ == "__main__":
    main()
