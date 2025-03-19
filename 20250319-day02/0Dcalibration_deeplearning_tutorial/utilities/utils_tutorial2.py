import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns


def get_config(mode="classification"):
    """
    Returns the configuration dictionary with the selected mode.

    Parameters:
        mode (str): "classification" or "regression"

    Returns:
        dict: Configuration settings
    """
    if mode not in ["classification", "regression"]:
        raise ValueError("Invalid mode! Choose 'classification' or 'regression'.")
    # Print the mode
    print(f"Running in {mode} mode.")

    return mode

def downsample_signal(signal, old_sampling_rate, sampling_rate):
    """
    Downsamples a given signal from old_fs to new_fs.
    
    Parameters:
        signal (numpy array): The original signal to downsample.
        old_fs (int): Original sampling frequency.
        new_fs (int): New sampling frequency.

    Returns:
        numpy array: The downsampled signal.
    """
    ds_factor = int(np.round(old_sampling_rate / sampling_rate))
    return signal[::ds_factor]  # Simple downsampling by selecting every nth sample

def load_all_ppg_radial(pwdb_data, old_sampling_rate, sampling_rate):
    """
    Loads and downsamples all PPG Radial signals for all subjects.

    Args:
        pwdb_data (dict): Preloaded PWDB data from a .mat file.
        new_fs (int, optional): Target downsampling frequency (default: 50 Hz).

    Returns:
        pd.DataFrame: PPG signals and metadata with uniform sampling rate.
    """
    if pwdb_data is None:
        raise ValueError("The 'pwdb_data' parameter must be provided.")

    # Extract PPG Radial signals
    try:
        PPG_Radial = pwdb_data['waves'][0, 0]['PPG_Radial'][0,0]  # originally MATLAB cell array (1 x N subjects)
        total_subjects = PPG_Radial.shape[1]  # Number of subjects
    except KeyError:
        raise ValueError("PPG_Radial field not found in dataset.")

    # Extract other relevant data
    age = pwdb_data['config'][0, 0]['age'][0,0].flatten()

    # Extract plausibility log
    plausibility_log = pwdb_data['plausibility'][0, 0]['plausibility_log'][0, 0].flatten()

    # Prepare data storage
    data_list = []

    for subject_id in range(total_subjects):
        ppg_signal = PPG_Radial[0, subject_id - 1].flatten()  # Extracting subject data (python indexing starts at 0)
        downsampled_signal = downsample_signal(ppg_signal, old_sampling_rate, sampling_rate)

        time_vector = np.arange(len(downsampled_signal)) / sampling_rate  # New time vector

        data_list.append({
            "subject_id": subject_id, 
            "age": age[subject_id],
            "fs": sampling_rate,
            "signal_length": len(downsampled_signal),
            "ppg_signal": downsampled_signal.tolist(),
            "time_vector": time_vector.tolist(),
            "plausibility_log": bool(plausibility_log[subject_id])  # Convert to boolean
        })

    df = pd.DataFrame(data_list)
    print(f"Loaded and downsampled PPG Radial signals for {total_subjects} subjects (from {old_sampling_rate}Hz to {sampling_rate}Hz).")

    return df


def split_train_test_data(dataset, mode="classification"):
    """
    Splits the dataset into training and testing sets based on age groups and physiological plausibility.
    
    Note: This method follows a specific structured splitting approach where:
    - Even-indexed subjects go into the training set.
    - Odd-indexed subjects go into the testing set.
    - Physiologically plausible subjects aged 25 and 75 are specifically selected for classification.
    - This is not the standard approach; normally, datasets are split using randomized methods (e.g., sklearn's train_test_split).
    
    Parameters:
        dataset (dict): Processed dataset containing sequences and metadata.
        mode (str, optional): "classification" for age groups, "regression" for numerical age prediction.

    Returns:
        tuple: (train_data, test_data) - dictionaries containing training and testing datasets.
    """
    even_indices = np.arange(len(dataset['age'])) % 2 == 0  # Even elements for train, odd for test
    
    if mode == "classification":
        plausibility_mask = dataset['plausibility_log'] == True  # Filter only plausible subjects
        
        young_indices = np.where((dataset['age'] == 25) & plausibility_mask)[0]
        elderly_indices = np.where((dataset['age'] == 75) & plausibility_mask)[0]
        
        # Splitting young and elderly subjects evenly between training and testing sets
        # We first extract the numeric indices where young or elderly subjects exist.
        # Then we select every second subject using slicing with different starting points:
        # - The training set picks indices starting from 0 (even indices: 0, 2, 4, ...)
        # - The testing set picks indices starting from 1 (odd indices: 1, 3, 5, ...)
        young_train = young_indices[0:len(young_indices):2]  # Select every 2nd young subject (starting at index 0)
        young_test = young_indices[1:len(young_indices):2]  # Select every 2nd young subject (starting at index 1)
        elderly_train = elderly_indices[0:len(elderly_indices):2]  # Select every 2nd elderly subject (starting at index 0)
        elderly_test = elderly_indices[1:len(elderly_indices):2]  # Select every 2nd elderly subject (starting at index 1)
        
        train_indices = np.concatenate((young_train, elderly_train))
        test_indices = np.concatenate((young_test, elderly_test))
    
    elif mode == "regression":
        valid_indices = dataset['age'] > 20
        train_indices = np.where(even_indices & valid_indices)[0]
        test_indices = np.where(~even_indices & valid_indices)[0]
    else:
        raise ValueError("Invalid mode! Choose 'classification' or 'regression'.")
    
    # Convert training dataset into a DataFrame
    train_data = pd.DataFrame({
        'ppg_signal': [dataset['ppg_signal'][i] for i in train_indices],
        'age': pd.Categorical(dataset['age'][train_indices]) if mode == "classification" else dataset['age'][train_indices],
    })
    
    # Convert testing dataset into a DataFrame
    test_data_full = pd.DataFrame({
        'subject_id': dataset['subject_id'][test_indices], #saving to explain potential misclassifications
        'ppg_signal': [dataset['ppg_signal'][i] for i in test_indices],
        'age': pd.Categorical(dataset['age'][test_indices]) if mode == "classification" else dataset['age'][test_indices],
    })
    
    test_data = test_data_full.drop(['subject_id'],axis=1)

    if mode == "classification":
        print(f"Training data: {len(train_data['age'])} subjects ({sum(train_data['age'] == 25)} young and {sum(train_data['age'] == 75)} elderly)")
        print(f"Testing data: {len(test_data['age'])} subjects ({sum(test_data['age'] == 25)} young and {sum(test_data['age'] == 75)} elderly)")
    else:
        print(f"Training data: {len(train_data['age'])} subjects")
        print(f"Testing data: {len(test_data['age'])} subjects")
    return train_data, test_data, test_data_full

def plot_pulse_waves(train_data, sampling_rate=50):
    """
    Plots an example of young and elderly PPG pulse waves from the dataset.

    Args:
        train_data (pd.DataFrame): Training dataset containing 'ppg_signal' and 'age'.
        sampling_rate (int, optional): Sampling frequency in Hz (default: 50 Hz).
    """
    padding_length = 2  # Number of samples to pad before and after

    # Find a young subject (age 25)
    young_subject = train_data[train_data['age'] == 25].iloc[1]  # A young subject
    young_ppg = np.concatenate((
        young_subject['ppg_signal'][-padding_length:],  # Append last samples at the start
        young_subject['ppg_signal'],  # Original PPG signal
        young_subject['ppg_signal'][:padding_length]  # Append first samples at the end
    )) + 0.6  # Offset to visually separate the signals

    # Find an elderly subject (age 75)
    elderly_subject = train_data[train_data['age'] == 75].iloc[1]  # An elderly subject
    elderly_ppg = np.concatenate((
        elderly_subject['ppg_signal'][-padding_length:],  # Append last samples at the start
        elderly_subject['ppg_signal'],  # Original PPG signal
        elderly_subject['ppg_signal'][:padding_length]  # Append first samples at the end
    ))

    # Generate time vectors for plotting
    time_young = np.arange(len(young_ppg)) / sampling_rate
    time_elderly = np.arange(len(elderly_ppg)) / sampling_rate

    # Plot the pulse waveforms
    plt.figure(figsize=(6, 6))
    plt.plot(time_young, young_ppg, 'b', linewidth=2, label='Young Subject')
    plt.plot(time_elderly, elderly_ppg, 'r', linewidth=2, label='Elderly Subject')

    # Formatting the plot
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('PPG Amplitude', fontsize=16)
    plt.ylim([-0.1, 1.7])
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks()
    plt.grid(False)

    plt.show()

def switch_to_tabular(df, column_name="ppg_signal", target_column="age", pad_value=-1):
    """
    Expands a time-series column into separate numbered columns while keeping the target column as last.

    Args:
        df (pd.DataFrame): DataFrame containing a column with time-series data.
        column_name (str): Name of the column to expand.
        target_column (str): Column to keep as the last column (e.g., 'age').

    Returns:
        pd.DataFrame: Expanded DataFrame with numbered time-step columns and 'age' as the last column.
    """
    # Expand time-series column into separate numbered columns
    ppg_expanded = pd.DataFrame(df[column_name].tolist())

    # Rename columns with ascending numbers
    ppg_expanded.columns = range(ppg_expanded.shape[1])

    # Extract the target column (age) and drop the original time-series column
    target_values = df[[target_column]].reset_index(drop=True)

    # Concatenate expanded data with target column at the end
    df_final = pd.concat([ppg_expanded, target_values], axis=1)

    return df_final

def scale_data(train_data, test_data):
    """
    Scales all numerical features in train and test datasets using MinMaxScaler, 
    except for the last column (target variable for regression).

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.

    Returns:
        tuple: (scaled_train_X, scaled_test_X, y_train, y_test, scaler_X, scaler_y) where:
            - scaled_train_X (pd.DataFrame): Scaled training features.
            - scaled_test_X (pd.DataFrame): Scaled testing features.
            - y_train (pd.Series): Target values (unscaled) for training.
            - y_test (pd.Series): Target values (unscaled) for testing.
            - scaler_X (MinMaxScaler): Scaler used for input features.
            - scaler_y (MinMaxScaler): Scaler used for target variable.
    """
    # Split features (X) and target (y)
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # Initialize MinMaxScaler for features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit and transform features
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    # Fit and transform target (for regression only)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    print("Data scaled using MinMaxScaler (features and target separately)")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def rescale_predictions(predictions, scaler_y):
    """
    Rescales model predictions back to the original scale.

    Args:
        predictions (numpy array): Model predictions.
        scaler_y (MinMaxScaler): Scaler used to normalize the target variable.

    Returns:
        numpy array: Rescaled predictions.
    """
    return scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

def padding(df, pad_value):
    """
    Replaces NaN values (from shorter sequences) with a specified padding value.

    Args:
        pad_value (int/float, optional): Value to replace NaNs for padding (default: -1).

    Returns:
        pd.DataFrame: Expanded DataFrame with numbered time-step columns and 'age' as the last column.
    """
    # Replace NaNs with the padding value (-1)
    df.fillna(pad_value, inplace=True)

    return df

def build_lstm_model(input_size=1, hidden_units=100, mode="classification"):
    """
    Builds an LSTM model for PPG classification or regression with proper masking.

    Args:
        input_size (int): Number of input features (1 for PPG).
        hidden_units (int): Number of LSTM hidden units.
        mode (str): "classification" (young vs. elderly) or "regression" (predict age).

    Returns:
        Keras Model: Compiled LSTM model.
    """
    model = Sequential()
    
    # Input layer with Masking to ignore padded values (-1)
    model.add(Input(shape=(None, input_size)))  # Accepts variable-length input
    model.add(Masking(mask_value=-1))  # Ignore padded values

    model.add(BatchNormalization())  # Normalization layer
    model.add(LSTM(hidden_units, return_sequences=False))  # LSTM layer

    if mode == "classification":
        model.add(Dense(2, activation='softmax'))  # 2 output neurons (Young vs. Elderly)
        loss = 'sparse_categorical_crossentropy'  # Classification loss
        metrics = ['accuracy']
    elif mode == "regression":
        model.add(Dense(1, activation=None))  # 1 output neuron (Predict Age)
        loss = 'mean_squared_error'  # Regression loss
        metrics = ['mae']  # Mean Absolute Error for evaluation

    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model

def get_training_options(train_data, mode="classification"):
    """
    Specifies training options for the LSTM model.

    Args:
        train_data (pd.DataFrame): The training dataset.
        mode (str): "classification" (young vs. elderly) or "regression" (predict age).

    Returns:
        dict: Training options (epochs, batch size, callbacks).
    """
    # Define epochs based on classification or regression
    max_epochs = 50 if mode == "classification" else 100

    # Compute batch size dynamically
    mini_batch_size = round(len(train_data) / (2 * 10))  # Equivalent to MATLAB miniBatchSize

    # Define training options (Keras doesn't need explicit execution environment setting)
    training_options = {
        "epochs": max_epochs,
        "batch_size": mini_batch_size,
        "callbacks": [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],  # Stop early if overfitting
        "verbose": 1  # Show training progress
    }

    return training_options

def plot_training_progress(history, mode):
    """
    Plots training loss and accuracy (or MAE for regression).

    Args:
        history (tf.keras.callbacks.History): Training history object.
        mode (str): "classification" or "regression".
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if mode == "classification" else 'Mean Absolute Error')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def train_lstm_model(model, train_data, mode="classification"):
    """
    Trains an LSTM model for PPG classification or regression, handling variable sequence lengths.

    Args:
        model (tf.keras.Model): Compiled LSTM model.
        train_data (pd.DataFrame): Training dataset.
        mode (str): "classification" (young vs. elderly) or "regression" (predict age).

    Returns:
        tf.keras.Model: Trained model.
        tf.keras.callbacks.History: Training history for visualization.
    """
    # Extract sequences (PPG signals) and labels
    X_train_raw = train_data['ppg_signal'].tolist()

    # Determine the max sequence length
    max_seq_length = max(len(seq) for seq in X_train_raw)

    # Pad sequences (using -1 as a padding marker)
    X_train = pad_sequences(X_train_raw, maxlen=max_seq_length, padding='post', dtype='float32', value=-1)

    # Convert labels to NumPy arrays
    y_train = np.array(train_data['age'])
    y_train = np.where(y_train == 25, 0, 1)  # Convert ages: 25 → 0, 75 → 1 ##TO DO ONLY IF CLASSIFICATION!!!
    
    # Expand dimensions to match LSTM input (batch, time steps, features)
    X_train = np.expand_dims(X_train, axis=-1)  # Shape: (samples, time_steps, features)

    # Training options
    train_options = get_training_options(train_data, mode=mode)

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=train_options["epochs"],
        batch_size=train_options["batch_size"],
        validation_split=0.2,
        callbacks=train_options["callbacks"],
        verbose=train_options["verbose"]
    )
    # Plot Training Loss & Accuracy
    plot_training_progress(history, mode)

    return model, history

def evaluate_model(model, test_data, scaler_y, mode="classification"):
    """
    Evaluates an LSTM model on the test dataset.

    Args:
        model (tf.keras.Model): Trained LSTM model.
        test_data (pd.DataFrame): Testing dataset.
        mode (str): "classification" (young vs. elderly) or "regression" (predict age).

    Returns:
        np.array: Model predictions.
    """
    # Extract test sequences
    X_test_raw = test_data['ppg_signal'].tolist()

    # Determine max sequence length from training
    max_seq_length = max(len(seq) for seq in X_test_raw)

    # Pad sequences for uniform length
    X_test = pad_sequences(X_test_raw, maxlen=max_seq_length, padding='post', dtype='float32', value=-1)

    # Expand dimensions to match LSTM input (batch, time steps, features)
    X_test = np.expand_dims(X_test, axis=-1)

    # Make predictions
    YPred_scaled = model.predict(X_test, batch_size=32)

    if mode == "classification":
        YPred = np.argmax(YPred_scaled, axis=1)  # Convert softmax probabilities to class labels

    else: # Rescaling predictions after regression
        Ypred = rescale_predictions(YPred_scaled, scaler_y)
    
    return YPred

def assess_model_performance(YPred, test_data, mode="classification"):
    """
    Assesses the performance of the trained LSTM model.

    Args:
        YPred (np.array): Model predictions.
        test_data (pd.DataFrame): Testing dataset.
        mode (str): "classification" (young vs. elderly) or "regression" (predict age).

    Returns:
        None: Prints evaluation metrics.
    """
    if mode == "classification":
        # Convert test labels to numerical format (0 for Young, 1 for Elderly)
        y_true = np.array(test_data['age'])
        y_true = np.where(y_true == 25, 0, 1)  # 25 → 0 (Young), 75 → 1 (Elderly)

        # Calculate accuracy
        acc = 100 * accuracy_score(y_true, YPred)
        print(f"Accuracy: {acc:.1f}%")
        print(f"Number of correctly classified PPG waves: {sum(YPred == y_true)} out of {len(YPred)}")
        
        # False negatives (elderly classified as young)
        fn_elderly_as_young = sum((YPred == 0) & (y_true == 1))
        print(f"🔹 Number of elderly PPG waves misclassified as young: {fn_elderly_as_young}")

        # False positives (young classified as elderly)
        fp_young_as_elderly = sum((YPred == 1) & (y_true == 0))
        print(f"🔹 Number of young PPG waves misclassified as elderly: {fp_young_as_elderly}")

        # Print a classification report
        print("\n📊 Classification Report:\n", classification_report(y_true, YPred, target_names=["Young", "Elderly"]))

        # Print confusion matrix
        cm = confusion_matrix(y_true, YPred)
        print("\n🔹 Confusion Matrix:\n", cm)

    else:  # Regression Mode
        y_true = np.array(test_data['age_num'])  # Numerical age labels

        # Compute bias and precision
        bias = np.mean(YPred - y_true)
        precision = 1.96 * np.std(YPred - y_true)

        print(f"Bias (Mean Error): {bias:.2f}")
        print(f"Precision (1.96 * STD): {precision:.2f}")

def analyze_classification_errors(YPred, test_data, pwdb_data): #ToDo only for classification mode
    """
    Identifies correct and incorrect classifications and plots boxplot for Cardiac Output (CO).

    Args:
        YPred (np.array): Model predictions.
        test_data (pd.DataFrame): Testing dataset with subject IDs.
        pwdb_data (dict): Original PWDB data containing haemodynamic parameters.

    Returns:
        None: Displays boxplot of CO values for correctly and incorrectly classified elderly subjects.
    """
    # Ensure 'subject_id' exists
    if 'subject_id' not in test_data:
        raise KeyError("Error: 'subject_id' is missing from test_data. Ensure it's included in dataset split.")

    # Convert test labels to numerical format (0 for Young, 1 for Elderly)
    y_true = np.array(test_data['age'])
    y_true = np.where(y_true == 25, 0, 1)  # 25 → 0 (Young), 75 → 1 (Elderly)

    # Find incorrect classifications
    incorrect_indices = np.where(YPred != y_true)[0]

    # Find correctly classified elderly subjects
    elderly_indices = np.where(y_true == 1)[0]  # True elderly labels
    correct_elderly_indices = np.setdiff1d(elderly_indices, incorrect_indices)

    # Extract Cardiac Output (CO) values from PWDB data
    co_values = np.array([pwdb_data['haemods'][0, 0]['CO'][0, subject - 1] for subject in test_data['subject_id']])

    # Extract CO values for correctly and incorrectly classified elderly subjects
    co_correct = np.array(co_values[correct_elderly_indices]).flatten()  # Ensure 1D
    co_incorrect = np.array(co_values[incorrect_indices]).flatten()  # Ensure 1D

    # Create a DataFrame for boxplot
    df = pd.DataFrame({
        "Cardiac Output (CO) (l/min)": np.concatenate([co_correct, co_incorrect]),  # Ensure 1D array
        "Classification": np.concatenate([np.full(len(co_correct), "Correct"), np.full(len(co_incorrect), "Incorrect")])
    })

    # Plot boxplot using seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Classification", y="Cardiac Output (CO) (l/min)", data=df, palette=["blue", "red"])

    # Formatting the plot
    plt.ylabel("CO (l/min)", fontsize=14)
    plt.xlabel("")
    plt.title("CO Distribution for Correct vs. Incorrect Classifications", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()
