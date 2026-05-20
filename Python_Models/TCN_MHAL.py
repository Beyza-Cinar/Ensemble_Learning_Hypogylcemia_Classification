# ===== Standard library =====
import os
import random
from collections import Counter

# ===== Numerical & data handling =====
import numpy as np
from numpy.random import default_rng
from scipy.special import softmax

# ===== Scikit-learn =====
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ===== Imbalanced-learn =====
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ===== TensorFlow / Keras =====
import tensorflow as tf
from tcn import TCN
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

tf.config.experimental.enable_op_determinism()

seed_value = 42
os.environ["PYTHONHASHSEED"] = str(seed_value)
tf.keras.utils.set_random_seed(seed_value)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("TF:", tf.__version__)
print("GPUs:", gpus)

#---------------- DL Models for Classification ----------------#
def flatten(subject):
    array_data = [np.array(x) for x in subject]  
    flattened_data = np.concatenate(array_data, axis=0)
    return flattened_data


def TCN_MH_model(timesteps = 25, n_features = 1 , num_classes = 4, nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=(1, 2, 4, 8, 16),  
        padding="same",
        use_skip_connections=True,
        dropout_rate=0.0, #0.1,
        use_batch_norm=False,
        use_layer_norm=True,
        activation='relu',
        return_sequences=True, 
        num_heads = 4):

    x_in = layers.Input(shape=(timesteps, n_features))

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,  
        padding=padding,
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate, #0.1,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        activation=activation,
        return_sequences=return_sequences,  
    )(x_in)

    # Self-attention over time
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=nb_filters// num_heads)(x, x)

    x = layers.GlobalAveragePooling1D()(x)

    y = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(x_in, y)
    return model 



class ThresholdController(BaseEstimator, TransformerMixin):
        def __init__(self, threshold):
            self.threshold = threshold

        def fit(self, X, y=None):
            return self

        def transform(self, X_dict):
            # Expect X_dict = {"X_seq": X, "model_pred": preds}
            X_seq = X_dict["X_seq"]
            preds = X_dict["model_pred"]

            # Last timestep
            last_val = X_seq[:, -1, 0]
            final_preds = preds.copy()

            final_preds[last_val <= self.threshold] = 0
            final_preds[(preds == 0) & (last_val > self.threshold)] = 1

            X_dict["final_pred"] = final_preds
            return X_dict


def data_initialization(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold):

    
    print(Y_train_fold.shape)
    # initiliazes buffer size and batch size 
    #BUFFER_SIZE = len(X_train_fold) 
    BATCH_SIZE = 2048 #512 #2048 #256
    BUFFER_SIZE = min(len(X_train_fold), 10_000)
    SEED = 42

    # for class weights 
    Y_train_weights = Y_train_fold.reshape(-1).astype('int32')

    print("original categroical encoding")
    Y_train_rs = to_categorical(Y_train_fold).astype(np.float32)
    Y_val_rs   = to_categorical(Y_val_fold).astype(np.float32)


    X_train_fold = X_train_fold.astype(np.float32, copy=False)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, Y_train_rs))
    train_ds = train_ds.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(1)

    X_val_fold = X_val_fold.astype(np.float32, copy=False)


    val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, Y_val_rs))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(1)

    return train_ds,val_ds, Y_train_weights, X_train_fold


def  TCN_initialization(Y_train_weights, timesteps=25, channels=1, n_classes=4, count =0):


    tf.keras.backend.clear_session()
    initial_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    #optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=initial_lr)
    # Define the early stopping callback
    early_stopping = EarlyStopping(patience=10, monitor="val_loss", mode="min", verbose=1, restore_best_weights=True)
    # initialize callbacks 
    reduceLR =ReduceLROnPlateau(monitor="val_loss", patience=5, verbose= 1, mode="min", factor= 0.2, min_lr = 1e-5) 

    checkpoint = ModelCheckpoint(
        f"MH_TCN_MHAL_Model_{count}.weights.h5", monitor="val_loss",
        verbose=1,save_weights_only=True, save_best_only=True, mode="min"
    )

    callbacks= [early_stopping, checkpoint, reduceLR]


    class_counts = np.bincount(Y_train_weights, minlength=4).astype(np.float32)
    inv_freq = 1.0 / np.maximum(class_counts, 1)
    alpha = inv_freq / np.mean(inv_freq)  # normalize mean≈1
    print("Alpha weights:", alpha)

    # Define focal loss with tuned alpha and gamma
    focloss = tf.keras.losses.CategoricalFocalCrossentropy(
        gamma=1.5,          # focus more on hard examples
        alpha=alpha,        # per-class weights
        )

    model_tcn_MH_model = TCN_MH_model(timesteps =timesteps, n_features = channels, num_classes = n_classes, nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=(1, 2, 4, 8, 16),  
        padding="same",
        use_skip_connections=True,
        dropout_rate=0.0, #0.1,
        use_batch_norm=False,
        use_layer_norm=True,
        activation='relu',
        return_sequences=True,  num_heads = 4) 
    model_tcn_MH_model.compile(loss=focloss, optimizer=optimizer, metrics=['categorical_accuracy'])
    #K.set_value(model_tcn.optimizer.learning_rate, initial_lr)
    
    return model_tcn_MH_model, callbacks



#------------------- MODELS --------------------#
def main():

    X_data, Y_data = load("Data/3Fold_Train_Data.joblib")

    # initializes 5 folds for cross validation
    kfolds = KFold(n_splits=3, shuffle=False)#, random_state=42)
    seed_value = 42

    count = 0

    # empty list for the predicted and actual values
    y_true_labels = []
    predicted_labels = []
    predicted_prob_labels = []

    # empty lists for the metrics calculated at each fold
    F1_macro = []
    Rec_macro = []
    Pre_macro = []
    F1_class = []
    Rec_class = []
    Pre_class = []
    PR_AUC_class = []
    PR_AVG_class = []
    PR_AUC_Mac = []
    PR_AVG_Mac = []

    for train_idx, test_idx in kfolds.split(X_data):

        # split into temp and test (20%)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.05, shuffle=False, random_state=seed_value)
        print(len(train_idx))
        print(len(val_idx))
        print(len(test_idx))

        average_precisions = []
        PRauc_scores = []

        print(f"Started for fold {count}")

        # extracts the selected subjects for training for the X and Y folds
        X_train_fold = [X_data[i] for i in train_idx]
        Y_train_fold = [Y_data[i] for i in train_idx]
        X_train_fold  = flatten(X_train_fold)
        Y_train_fold  = flatten(Y_train_fold)

        # extracts the selected subjects for validating for the X and Y folds
        X_val_fold = [X_data[i] for i in val_idx]
        Y_val_fold = [Y_data[i] for i in val_idx]
        X_val_fold  = flatten(X_val_fold)
        Y_val_fold  = flatten(Y_val_fold)

        X_test_fold = [X_data[i] for i in test_idx]
        Y_test_fold = [Y_data[i] for i in test_idx]
        # flattens the arrays of single subjects into one array of shape (X,25,2)
        X_test_fold = flatten(X_test_fold)
        # flattens the arrays of single subjects into one array of shape (X,1)
        Y_test_fold = flatten(Y_test_fold)

        Y_test_rs = to_categorical(Y_test_fold)
        X_test_t = tf.convert_to_tensor(X_test_fold, dtype=tf.float32)
        Y_test_t = tf.convert_to_tensor(Y_test_rs, dtype=tf.int32)
        Y_test_all = Y_test_fold.reshape(-1)


        train_ds,val_ds, Y_train_weights, X_train_fold = data_initialization(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold)
        n_classes= len(set(Y_train_weights))
        timesteps = X_train_fold.shape[1]
        channels = X_train_fold.shape[2]
        model_tcn_mh, callbacks = TCN_initialization(Y_train_weights, timesteps=timesteps, channels=channels, n_classes=n_classes,count=count)
        print("initialized successfully")
        print("started training")
        model_tcn_mh.fit(train_ds, epochs=100, verbose=1, validation_data=val_ds, callbacks=callbacks)
    
        
        y_pred_prob = model_tcn_mh.predict(X_test_t)
        y_pred_labels = np.argmax(y_pred_prob, axis=1) 
            
        y_true_labels.append(Y_test_fold)  
        predicted_prob_labels.append(y_pred_prob) 
        predicted_labels.append(y_pred_prob)


        f1_score_mac = f1_score(Y_test_all, y_pred_labels, average='macro')
        F1_macro.append(f1_score_mac)

        rec_scores_mac = recall_score(Y_test_all, y_pred_labels, average='macro')
        Rec_macro.append(rec_scores_mac)

        pre_scores_mac = precision_score(Y_test_all, y_pred_labels, average='macro')
        Pre_macro.append(pre_scores_mac)

        f1_scores = f1_score(Y_test_all, y_pred_labels, average=None, labels=np.arange(4))
        F1_class.append(f1_scores)

        rec_scores = recall_score(Y_test_all, y_pred_labels, average=None, labels=np.arange(4))
        Rec_class.append(rec_scores)

        pre_scores = precision_score(Y_test_all, y_pred_labels, average=None, labels=np.arange(4))
        Pre_class.append(pre_scores)

        classes = set(Y_test_all)
        for n_cl, cl in enumerate(classes):  
                f_pre, f_rec, t = precision_recall_curve(Y_test_rs[:, cl], y_pred_prob[:, cl])
                sorted_indices = np.argsort(f_rec)
                pr_auc_scores = auc(f_rec, f_pre)
                ap_score = average_precision_score(Y_test_rs[:, cl], y_pred_prob[:, cl])
                average_precisions.append(ap_score)
                PRauc_scores.append(pr_auc_scores)

        PR_AVG_class.append(average_precisions)

        PR_AUC_class.append(PRauc_scores)
        PRAUC_mean = np.mean(PRauc_scores)
        PR_AUC_Mac.append(PRAUC_mean)

        count = count + 1

        
    # Saves reports of all folds and macro averages into txt file
    with open("TCN_MH_3Fold.txt", "w") as f:

        f.write("Macro F1 Measure:")
        f.write(", ".join([f"{score:.4f}" for score in F1_macro]) + "\n")
        f.write(f"Mean Macro F1 Measure: {np.mean(F1_macro):.4f}" +"\n")
        f.write(f"STD Macro F1 Measure: {np.std(F1_macro):.4f}" +"\n\n")

        f.write("Macro Recall:")
        f.write(", ".join([f"{score:.4f}" for score in Rec_macro]) + "\n")
        f.write(f"Mean Macro Recall: {np.mean(Rec_macro):.4f}" +"\n")
        f.write(f"STD Macro Recall: {np.std(Rec_macro):.4f}" +"\n\n")

        f.write("Macro Precision:")
        f.write(", ".join([f"{score:.4f}" for score in Pre_macro]) + "\n")
        f.write(f"Mean Macro Precision: {np.mean(Pre_macro):.4f}" +"\n")
        f.write(f"STD Macro Precision: {np.std(Pre_macro):.4f}" +"\n\n")

        f.write("Macro PR AUC:")
        f.write(", ".join([f"{score:.4f}" for score in PR_AUC_Mac]) + "\n")
        f.write(f"Mean Macro PR AUC: {np.mean(PR_AUC_Mac):.4f}" +"\n")
        f.write(f"STD Macro PR AUC: {np.std(PR_AUC_Mac):.4f}" +"\n\n")

        f.write("Macro AVG PR:")
        f.write(", ".join([f"{score:.4f}" for score in PR_AVG_Mac]) + "\n")
        f.write(f"Mean Macro AVG PR: {np.mean(PR_AVG_Mac):.4f}" +"\n")
        f.write(f"STD Macro AVG PR: {np.std(PR_AVG_Mac):.4f}" +"\n\n")

        f.write("F1 Measure per Class:")
        f.write(", ".join([f"{score}" for score in F1_class]) + "\n\n")
    
        f.write("Recall per Class:")
        f.write(", ".join([f"{score}" for score in Rec_class]) + "\n\n")

        f.write("Precision per Class:")
        f.write(", ".join([f"{score}" for score in Pre_class]) + "\n\n")

        f.write("PR AUC per Class:")
        f.write(", ".join([f"{score}" for score in PR_AUC_class]) + "\n\n")



    # Save (single file)
    dump(y_true_labels, "y_true_TCN_MH.joblib", compress=("zlib", 1))
    dump(predicted_labels, "y_pred_TCN_MH.joblib", compress=("zlib", 1))
    dump(predicted_prob_labels, "y_pred_prob_TCN_MH.joblib", compress=("zlib", 1))

  

   
if __name__ == "__main__":
    main()

