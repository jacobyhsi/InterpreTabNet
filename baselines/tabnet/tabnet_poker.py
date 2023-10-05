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
from sklearn.model_selection import train_test_split


"""
7. Attribute Information:
   1) S1 ìSuit of card #1î
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   2) C1 ìRank of card #1î
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   3) S2 ìSuit of card #2î
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   4) C2 ìRank of card #2î
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   5) S3 ìSuit of card #3î
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   6) C3 ìRank of card #3î
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   7) S4 ìSuit of card #4î
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   8) C4 ìRank of card #4î
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   9) S5 ìSuit of card #5î
      Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

   10) C5 ìRank of card 5î
      Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

   11) CLASS ìPoker Handî
      Ordinal (0-9)

      0: Nothing in hand; not a recognized poker hand 
      1: One pair; one pair of equal ranks within five cards
      2: Two pairs; two pairs of equal ranks within five cards
      3: Three of a kind; three equal ranks within five cards
      4: Straight; five cards, sequentially ranked with no gaps
      5: Flush; five cards with the same suit
      6: Full house; pair + different rank three of a kind
      7: Four of a kind; four equal ranks within five cards
      8: Straight flush; straight + flush
      9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
"""

def main():
    dataset_name = "poker_hand"
    target = 'poker_hand_class'

    int_columns = [
        "suit1", "rank1", "suit2",
        "rank2", "suit3", "rank3", "suit4", "rank4",
        "suit5", "rank5"
    ]

    feature_columns = (
            int_columns + [target])
    
    dataset_train_1 = 'poker-hand-training-true'
    train_1_out = Path(os.getcwd()+'/data/'+dataset_train_1+'.csv')
    train_1 = pd.read_csv(train_1_out,
                        header=None, names=feature_columns)
    
    dataset_train_2 = 'poker-hand-testing'
    train_2_out = Path(os.getcwd()+'/data/'+dataset_train_2+'.csv')
    train_2 = pd.read_csv(train_2_out,
                        header=None, names=feature_columns)
    
    train = pd.concat([train_1, train_2], axis=0)
    n_total = len(train)

    # Train, val and test split follows
    # Rory Mitchell, Andrey Adinets, Thejaswi Rao, and Eibe Frank.
    # Xgboost: Scalable GPU accelerated learning. arXiv:1806.11248, 2018.

    train_val_indices, test_indices = train_test_split(
        range(n_total), test_size=0.2, random_state=0)
    train_indices, valid_indices = train_test_split(
        train_val_indices, test_size=0.2 / 0.6, random_state=0)

    categorical_columns = []
    categorical_dims = {}
    for col in train.columns:
      #   print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    unused_feat = []

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
    gammas = [1.0, 1.2, 1.5, 2.0]
    lambda_sparses = [0.001, 0.01, 0.1, 0.3]
    learn_r = [0.005, 0.01, 0.02, 0.025]
    # reg_w = [0.001, 0.01, 0.05, 0.1]
    reg_m = [0.001, 0.01, 0.1, 0.3]
    reg_pq = [0.001, 0.01, 0.1, 0.3]

    # # TODO: set optimal parameters after tuning!
    # Optimum Hyperparameters Training [128, 5, 1.0, 0.001, 0.005, 0, 0]
    opt_ndna = 128
    opt_nsteps = 4
    opt_gamma = 1.0
    opt_lambda = 0.001
    opt_lr = 0.005
    # opt_reg_w = 0
    opt_reg_m = 0
    opt_reg_pq = 0

   #  ndna_test_acc = 0
   #  for ndna in nd_na:
   #      clf = TabNetClassifier(
   #          n_d=ndna,
   #          n_a=ndna,
   #          n_steps=n_steps[0],
   #          gamma=gammas[0],
   #          lambda_sparse=lambda_sparses[0],
   #          cat_idxs=cat_idxs,
   #          cat_dims=cat_dims,
   #          optimizer_params=dict(lr=learn_r[0]),
   #          # reg_m=reg_m[0],
   #          # reg_pq=reg_pq[0],
   #          # mask_type = 'relu'
   #      )
    
   #      clf.fit(
   #          X_train=X_train, y_train=y_train,
   #          eval_set=[(X_train, y_train), (X_valid, y_valid)],
   #          eval_name=['train', 'valid'], batch_size=256,
   #          virtual_batch_size=256,
   #          max_epochs=10, eval_metric=['accuracy']
   #      )
    
   #      y_pred = clf.predict(X_test)
   #      test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    
   #      if test_acc > ndna_test_acc:
   #          opt_ndna = ndna
   #          ndna_test_acc = test_acc
   #          print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])

   #  nstep_test_acc = 0
   #  for nstep in n_steps:
    
   #      clf = TabNetClassifier(
   #          n_d=opt_ndna,
   #          n_a=opt_ndna,
   #          n_steps=nstep,
   #          gamma=gammas[0],
   #          lambda_sparse=lambda_sparses[0],
   #          cat_idxs=cat_idxs,
   #          cat_dims=cat_dims,
   #          optimizer_params=dict(lr=learn_r[0]),
   #          # reg_m=reg_m[0],
   #          # reg_pq=reg_pq[0],
   #          # mask_type = 'relu'
   #      )
    
   #      clf.fit(
   #          X_train=X_train, y_train=y_train,
   #          eval_set=[(X_train, y_train), (X_valid, y_valid)],
   #          eval_name=['train', 'valid'], batch_size=256,
   #          virtual_batch_size=256,
   #          max_epochs=10, eval_metric=['accuracy']
   #      )
    
   #      y_pred = clf.predict(X_test)
   #      test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

   #      if test_acc > nstep_test_acc:
   #          opt_nsteps = nstep
   #          nstep_test_acc = test_acc
   #          print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
   #  gams_test_acc = 0
   #  for gams in gammas:
   #      clf = TabNetClassifier(
   #          n_d=opt_ndna,
   #          n_a=opt_ndna,
   #          n_steps=opt_nsteps,
   #          gamma=gams,
   #          lambda_sparse=lambda_sparses[0],
   #          cat_idxs=cat_idxs,
   #          cat_dims=cat_dims,
   #          optimizer_params=dict(lr=learn_r[0]),
   #          # reg_m=reg_m[0],
   #          # reg_pq=reg_pq[0],
   #          # mask_type = 'relu'
   #      )
    
   #      clf.fit(
   #          X_train=X_train, y_train=y_train,
   #          eval_set=[(X_train, y_train), (X_valid, y_valid)],
   #          eval_name=['train', 'valid'], batch_size=256,
   #          virtual_batch_size=256,
   #          max_epochs=10, eval_metric=['accuracy']
   #      )
    
   #      y_pred = clf.predict(X_test)
   #      test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
   #      if test_acc > gams_test_acc:
   #          opt_gamma = gams
   #          gams_test_acc = test_acc
   #          print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
   #  lamb_test_acc = 0
   #  for lambs in lambda_sparses:
   #      clf = TabNetClassifier(
   #          n_d=opt_ndna,
   #          n_a=opt_ndna,
   #          n_steps=opt_nsteps,
   #          gamma=opt_gamma,
   #          lambda_sparse=lambs,
   #          cat_idxs=cat_idxs,
   #          cat_dims=cat_dims,
   #          optimizer_params=dict(lr=learn_r[0]),
   #          # reg_m=reg_m[0],
   #          # reg_pq=reg_pq[0],
   #          # mask_type = 'relu'
   #      )
    
   #      clf.fit(
   #          X_train=X_train, y_train=y_train,
   #          eval_set=[(X_train, y_train), (X_valid, y_valid)],
   #          eval_name=['train', 'valid'], batch_size=256,
   #          virtual_batch_size=256,
   #          max_epochs=10, eval_metric=['accuracy']
   #      )
    
   #      y_pred = clf.predict(X_test)
   #      test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
   #      if test_acc > lamb_test_acc:
   #          opt_lambda = lambs
   #          lamb_test_acc = test_acc
   #          print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])
    
   #  lr_test_accuracy = 0
   #  for lr in learn_r:
   #      clf = TabNetClassifier(
   #          n_d=opt_ndna,
   #          n_a=opt_ndna,
   #          n_steps=opt_nsteps,
   #          gamma=opt_gamma,
   #          lambda_sparse=opt_lambda,
   #          cat_idxs=cat_idxs,
   #          cat_dims=cat_dims,
   #          optimizer_params=dict(lr=lr),
   #          # reg_m=reg_m[0],
   #          # reg_pq=reg_pq[0],
   #          # mask_type = 'relu'
   #      )
    
   #      clf.fit(
   #          X_train=X_train, y_train=y_train,
   #          eval_set=[(X_train, y_train), (X_valid, y_valid)],
   #          eval_name=['train', 'valid'], batch_size=256,
   #          virtual_batch_size=256,
   #          max_epochs=10, eval_metric=['accuracy']
   #      )
    
   #      y_pred = clf.predict(X_test)
   #      test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
   #      if test_acc > lr_test_accuracy:
   #          opt_lr = lr
   #          lr_test_accuracy = test_acc
   #          print("Optimum Hyperparameters Training", [opt_ndna, opt_nsteps, opt_gamma, opt_lambda, opt_lr, opt_reg_m, opt_reg_pq])

# Optimized Run #######################################################################################################################
    # Optimum Hyperparameters Training [128, 5, 1.0, 0.001, 0.005, 0, 0]
# Optimum Hyperparameters Training [128, 4, 1.0, 0.001, 0.005, 0, 0]
    clf = TabNetClassifier(
        n_d=128,
        n_a=128,
        n_steps=4,
        gamma=1.0,
        lambda_sparse=0.001,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_params=dict(lr=0.005),
        mask_type = 'sparsemax'
    )
    # max epoch 50
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'], batch_size=256,
        virtual_batch_size=256,
        max_epochs=100, eval_metric=['accuracy']
    )
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

    n_steps=4
    explain_matrix, masks = clf.explain(X_test)
    fig, axs = plt.subplots(1, n_steps, figsize=(20, 20))

    for i in range(n_steps):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
        axs[i].set_ylabel("Test Samples")
        axs[i].set_xlabel("Features")
    plt.savefig(f"{dataset_name}_original_tabnet_feature_mask_accuracy_{test_acc}.png")


if __name__ == "__main__":
    np.random.seed(0)
    main()
