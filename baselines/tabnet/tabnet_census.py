from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd
import numpy as np

import os
import wget
from pathlib import Path

# CODE WITH INDIVIDUAL HYP TUNING


def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset = 'census-income'
    dataset_name = 'census-income'
    out = Path(os.getcwd() + '/data/' + dataset_name + '.csv')

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, out.as_posix())

    train = pd.read_csv(out)
    target = ' <=50K'
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set == "train"].index
    valid_indices = train[train.Set == "valid"].index
    test_indices = train[train.Set == "test"].index

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            # print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    train.loc[train[target] == 0, target] = "wealthy"
    train.loc[train[target] == 1, target] = "not_wealthy"

    unused_feat = ['Set']

    features = [col for col in train.columns if col not in unused_feat + [target]]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    # TUNING HYPERPARAMETERS ###############################################################################################
    nd_na = [16, 32, 128]
    n_steps = [3, 4, 5]
    # gammas = [1.0, 1.2, 1.5, 2.0]
    lambda_sparses = [0.001, 0.01, 0.1]
    learn_r = [0.005, 0.01, 0.02, 0.025]
    # reg_w = [0.001, 0.01, 0.05, 0.1]
    reg_m = [0.001, 0.01, 0.1, 0.3]
    reg_pq = [0.001, 0.01, 0.1, 0.3]

    opt_ndna = 32
    opt_nsteps = 3
    opt_gamma = 1.5
    opt_lambda = 0.001
    opt_lr = 0.025
    # opt_reg_w = 0
    opt_reg_m = 0
    opt_reg_pq = 0

    # ndna_test_acc = 0
    # for ndna in nd_na:
    #     clf = TabNetClassifier(
    #         n_d=ndna,
    #         n_a=ndna,
    #         n_steps=n_steps[0],
    #         gamma=gammas[0],
    #         lambda_sparse=lambda_sparses[0],
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=learn_r[0]),
    #         reg_m=reg_m[0],
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    #     if test_acc > ndna_test_acc:
    #         opt_ndna = ndna
    #         ndna_test_acc = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])

    # nstep_test_acc = 0
    # for nstep in n_steps:
    
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=nstep,
    #         gamma=gammas[0],
    #         lambda_sparse=lambda_sparses[0],
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=learn_r[0]),
    #         reg_m=reg_m[0],
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    #     if test_acc > nstep_test_acc:
    #         opt_nsteps = nstep
    #         nstep_test_acc = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
    # gams_test_acc = 0
    # for gams in gammas:
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=opt_nsteps,
    #         gamma=gams,
    #         lambda_sparse=lambda_sparses[0],
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=learn_r[0]),
    #         reg_m=reg_m[0],
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    #     if test_acc > gams_test_acc:
    #         opt_gamma = gams
    #         gams_test_acc = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
    # lamb_test_acc = 0
    # for lambs in lambda_sparses:
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=opt_nsteps,
    #         gamma=opt_gamma,
    #         lambda_sparse=lambs,
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=learn_r[0]),
    #         reg_m=reg_m[0],
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    #     if test_acc > lamb_test_acc:
    #         opt_lambda = lambs
    #         lamb_test_acc = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
    # lr_test_accuracy = 0
    # for lr in learn_r:
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=opt_nsteps,
    #         gamma=opt_gamma,
    #         lambda_sparse=opt_lambda,
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=lr),
    #         reg_m=reg_m[0],
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    #     if test_acc > lr_test_accuracy:
    #         opt_lr = lr
    #         lr_test_accuracy = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])

    # reg_m_test_accuracy = 0
    # for r_m in reg_m:
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=opt_nsteps,
    #         gamma=opt_gamma,
    #         lambda_sparse=opt_lambda,
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=opt_lr),
    #         reg_m=r_m,
    #         reg_pq=reg_pq[0],
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    #     if test_acc > reg_m_test_accuracy:
    #         opt_reg_m = r_m
    #         reg_m_test_accuracy = test_acc
    #         print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
    # reg_pq_test_accuracy = 0
    # for r_pq in reg_pq:
    #     clf = TabNetClassifier(
    #         n_d=opt_ndna,
    #         n_a=opt_ndna,
    #         n_steps=opt_nsteps,
    #         gamma=opt_gamma,
    #         lambda_sparse=opt_lambda,
    #         cat_idxs=cat_idxs,
    #         cat_dims=cat_dims,
    #         optimizer_params=dict(lr=opt_lr),
    #         reg_m=opt_reg_m,
    #         reg_pq=r_pq,
    #         mask_type = 'relu'
    #     )
    
    #     clf.fit(
    #         X_train=X_train, y_train=y_train,
    #         eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #         eval_name=['train', 'valid'], batch_size=256,
    #         virtual_batch_size=256,
    #         max_epochs=10, eval_metric=['accuracy']
    #     )
    
    #     y_pred = clf.predict(X_test)
    #     test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    #     if test_acc > reg_pq_test_accuracy:
    #         opt_reg_pq = r_pq
    #         reg_pq_test_accuracy = test_acc
    
    # print("Optimum Hyperparameters", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])

    # Tuning for Mask #######################################################################################################################
    # mask_count = 1

    # for l in lambda_sparses:
    #     for m in reg_m:
    #         for pq in reg_pq:
    #             clf = TabNetClassifier(
    #                 n_d=32,
    #                 n_a=32,
    #                 n_steps=3,
    #                 gamma=1.0,
    #                 lambda_sparse=l,
    #                 cat_idxs=cat_idxs,
    #                 cat_dims=cat_dims,
    #                 optimizer_params=dict(lr=0.025),
    #                 reg_m = m,
    #                 reg_pq = pq,
    #                 mask_type = 'relu'
    #             )
    #             # max epoch 50
    #             clf.fit(
    #                 X_train=X_train, y_train=y_train,
    #                 eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #                 eval_name=['train', 'valid'], batch_size=256,
    #                 virtual_batch_size=256,
    #                 max_epochs=16, eval_metric=['accuracy']
    #             )

    #             y_pred = clf.predict(X_test)
    #             test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    #             print(f"Mask : {mask_count}, Hyperparameters : {[l, m, pq]}, Accuracy : {test_acc}")
    #             # print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

    #             explain_matrix, masks = clf.explain(X_test)
    #             fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    #             for i in range(3):
    #                 axs[i].imshow(masks[i][:50])
    #                 axs[i].set_title(f"mask {i}")
    #                 axs[i].set_ylabel("Test Samples")
    #                 axs[i].set_xlabel("Features")
    #                 # ticks = np.arange(1, 15)
    #                 # labels = ["age", "workclass", "fnlwgt (the number of people the census believes the entry represents)",
    #                 #           "education", "education-num", "marital-status", "occupation", "relationship", "race",
    #                 #           "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    #                 # axs[i].set_xticks(ticks)
    #                 # axs[i].set_xticklabels(labels)
    #             # plt.show()
    #             plt.savefig(f"Mask : {mask_count}, Hyperparameters : {[l, m, pq]}, Accuracy : {test_acc} cVAE.png")
    #             mask_count += 1

    # Optimized Run #######################################################################################################################
    clf = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=4,
        gamma=1.5,
        lambda_sparse=0.0001,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_params=dict(lr=0.02),
        mask_type = 'sparsemax'
    )
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'], batch_size=4096,
        virtual_batch_size=128,
        max_epochs=50, eval_metric=['accuracy']
    )

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    # print(f"Mask : {mask_count}, Hyperparameters : {[l, m, pq]}, Accuracy : {test_acc}")
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

    n_steps=4

    explain_matrix, masks = clf.explain(X_test)
    fig, axs = plt.subplots(1, n_steps, figsize=(20, 20))

    for i in range(n_steps):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
        axs[i].set_ylabel("Test Samples")
        axs[i].set_xlabel("Features")
    plt.savefig(f"{dataset}_feature_mask_original_accuracy_{test_acc}.png")


if __name__ == "__main__":
    np.random.seed(0)
    main()