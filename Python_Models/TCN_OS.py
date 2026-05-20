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

def TCN_model(timesteps = 25, n_features = 1 , num_classes = 4, nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=(1, 2, 4, 8, 16),  
        padding="same",
        use_skip_connections=True,
        dropout_rate=0.0, #0.1,
        use_batch_norm=False,
        use_layer_norm=True,
        activation='relu',
        return_sequences=False):

    x_in = layers.Input(shape=(timesteps, n_features))

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,  # enough for T=25, i need at least 4 so 8
        padding=padding,
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate, #0.1,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        activation=activation,
        return_sequences=return_sequences,  
    )(x_in)

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


def data_initialization(X_train_fold, Y_train_fold):

    print(Y_train_fold.shape)
    # initiliazes buffer size and batch size 
    #BUFFER_SIZE = len(X_train_fold) 
    BATCH_SIZE = 2048 #512 #2048 #256
    BUFFER_SIZE = min(len(X_train_fold), 10_000)
    SEED = 42 
    Y_train_rs = to_categorical(Y_train_fold)

    # for class weights 
    Y_train_weights = Y_train_fold.reshape(-1).astype('int32')

    print("original categroical encoding")
    Y_train_rs = to_categorical(Y_train_fold)
    
    
    
    # converts train, val, and test data into tensors

    X_train_fold = X_train_fold.astype(np.float32, copy=False)
    Y_train_rs   = Y_train_rs.astype(np.float32, copy=False)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, Y_train_rs))
    train_ds = train_ds.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(1)

   
    return train_ds,Y_train_weights, X_train_fold


def  TCN_initialization(Y_train_weights, timesteps=25, channels=1, n_classes=4, count=0):


    tf.keras.backend.clear_session()
    initial_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    # Define the early stopping callback
    early_stopping = EarlyStopping(patience=10, monitor="val_loss", mode="min", verbose=1, restore_best_weights=True)
    # initialize callbacks 
    reduceLR =ReduceLROnPlateau(monitor="val_loss", patience=5, verbose= 1, mode="min", factor= 0.2, min_lr = 1e-5) 

    checkpoint = ModelCheckpoint(
        f"OS_MH_TCN_Norm_Model_{count}.weights.h5", monitor="val_loss",
        verbose=1, save_best_only=True, mode="min"
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

    model_tcn = TCN_model(timesteps = timesteps, n_features = channels, num_classes = n_classes , nb_filters=64,
            kernel_size=3,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16),  
            padding="same",
            use_skip_connections=True,
            dropout_rate=0.0, #0.1,
            use_batch_norm=False,
            use_layer_norm=True,
            activation='relu',
            return_sequences=False) 
    model_tcn.compile(loss=focloss, optimizer=optimizer, metrics=['categorical_accuracy'])
    #K.set_value(model_tcn.optimizer.learning_rate, initial_lr)
    
    return model_tcn



#------------------- MODELS --------------------#
def main():

    
    X_data, Y_data = load("Data/3Fold_Train_Data.joblib")
    
    print("Data loaded successfully")
    
    X_data = flatten(X_data)
    Y_data = flatten(Y_data)


    train_ds,Y_train_weights, X_train_fold = data_initialization(X_data, Y_data)
    n_classes= len(set(Y_train_weights))
    timesteps = X_train_fold.shape[1]
    channels = X_train_fold.shape[2]
    model_tcn= TCN_initialization(Y_train_weights, timesteps=timesteps, channels=channels, n_classes=n_classes)
    print("initialized successfully")
    print("started training")
    model_tcn.fit(train_ds, epochs=35, verbose=2)
    name_tcn = "TCN_model_OS.keras"
    model_tcn.save(name_tcn)
    print("model saved")


if __name__ == "__main__":
    main()
