
import os
import shutil
import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import sys

#....CHANGED...........................................................................................................
if "check_bd_clicked" not in st.session_state:
    st.session_state["check_bd_clicked"] = False
if "bd_output" not in st.session_state:
    st.session_state["bd_output"] = ""



# Set a random seed for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Define the main folder path
MAINFOLDER = r"./APPdata_PS1C1"

# Create other paths relative to the main folder
training_file_path = os.path.join(MAINFOLDER, "Training", "Training_PS1C1.xlsx")  # FIXED TRAINING DATA
test_file_path = os.path.join(MAINFOLDER, "24hrData", "Dailydata_PS1C1.xlsx")  # DAILY DATA
excel_file_path = os.path.join(MAINFOLDER, "Breakdownrecords_PS1C1.xlsx")  # Recording excel for BD
folderpath = os.path.join(MAINFOLDER, "TemporaryData")  # Temporary dump files collector
threshold_file_path = os.path.join(MAINFOLDER,"Thresholds_PS1C1.xlsx") #

# Define the path to save models within the main folder
model_folder_path = os.path.join(MAINFOLDER, "Models")


uploaded_files = []  # List to keep track of uploaded files

# Streamlit UI
st.title("Breakdown Predictor")
st.markdown("Upload your files, and they will be preprocessed accordingly.")


# Initialize file uploader key in session state
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# File uploader
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, key=str(st.session_state["file_uploader_key"]))


# Show status
status_placeholder = st.empty()

# Function to clear old files from the folder
def clear_saved_files():
    try:
        # Clear old files in the folder
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove the file
            except Exception as e:
                status_placeholder.error(f"Error clearing old files: {e}")
                return
        status_placeholder.success("Saved files cleared successfully!")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to handle file saving (clear old files before saving new ones)
def save_files(uploaded_files):
    try:
        if not uploaded_files:
            status_placeholder.error("No files to save!")
            return

        # Clear old files in the folder before saving new files
        clear_saved_files()

        # Save each file from the uploaded list to the target folder
        for file in uploaded_files:
            with open(os.path.join(folderpath, file.name), "wb") as f:
                f.write(file.getbuffer())

        status_placeholder.success("Files saved successfully!")
        # Clear uploaded files from the interface after saving   addedd extra
        st.session_state["file_uploader_key"] += 1


    except Exception as e:
        status_placeholder.error(f"Error: {e}")



# Clear previous uploaded files display automatically before handling new uploads
if st.button("Save Files"):
    if uploaded_files:
        st.session_state['uploaded_files'] = None  # Reset session state to clear display
        st.session_state['uploaded_files'] = uploaded_files  # Store new uploads in session state
        save_files(st.session_state['uploaded_files'])  # Clear old files and save new ones
    else:
        st.error("Please upload files first.")



################# data preprocessing         ###################


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def process_data():
    
    # Define the input file (only one file in the folder)
    input_file_name = os.listdir(folderpath)[0]  # Assuming only one file in the folder
    input_file_path = os.path.join(folderpath, input_file_name)

    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        st.error(f"Input file '{input_file_name}' does not exist!")
        return

    # List of 12 unique asset names
    assets_list = [
        "PS-1 Primer Coat Exhaust Blower-Motor DE","PS-1 Primer Coat Exhaust Blower-Motor NDE","PS-1 Top Coat Exhaust Blower- Blower DE",
        "PS-1 Top Coat Exhaust Blower- Blower NDE","PS-1 Top Coat Exhaust Blower- Motor DE", "PS-1 Top Coat Exhaust Blower-Motor NDE ",
        ]

    # Columns to extract for each asset, corresponding to F, I, L, O, R, U
    required_column_indices = [5, 8, 11, 14, 17]  # 0-based indices for F, I, L, O, R, U
    required_column_names = ['a2', 'vv2', 'av2', 'hv2', 't2']


    # Load the input file
    input_df = pd.read_excel(input_file_path)

    # Initialize an empty DataFrame to store combined data
    output_df = pd.DataFrame()

    # Loop over each asset in assets_list
    for asset_name in assets_list:
        # Find rows where Column B (index 1) matches the asset_name
        asset_rows = input_df[input_df.iloc[:, 1] == asset_name]
        
        # Check if any rows were found
        if not asset_rows.empty:
            # Parse the date and time from Column C (index 2)
            asset_rows['DateTime'] = pd.to_datetime(asset_rows.iloc[:, 2], format='%d-%m-%Y %H:%M')

            
            
            # Identify the earliest start time in the data for this asset
            start_time = asset_rows['DateTime'].min().replace(hour=5, minute=30)
            end_time = start_time + timedelta(days=1, hours=0, minutes=0)
            
            # Filter rows within this 24-hour window (from earliest 5:30 AM to the next day 5:30 AM)
            filtered_rows = asset_rows[(asset_rows['DateTime'] >= start_time) & (asset_rows['DateTime'] <= end_time)]
            
            # Select only the first 49 rows if there are more than 49 available
            filtered_rows = filtered_rows.head(49)
            
            # Collect only the specified columns (F, I, L, O, R, U) for the 49 rows
            data_for_asset = filtered_rows.iloc[:, required_column_indices].values
            data_for_asset = pd.DataFrame(data_for_asset, columns=required_column_names)
            
            # Fill any missing rows with 0s if there are fewer than 49 rows
            if len(data_for_asset) < 49:
                missing_rows = 49 - len(data_for_asset)
                data_for_asset = pd.concat([data_for_asset, pd.DataFrame(0, index=range(missing_rows), columns=required_column_names)], ignore_index=True)
        else:
            # If no rows found for this asset, fill with 0s for all columns
            data_for_asset = pd.DataFrame(0, index=range(49), columns=required_column_names)

        # Rename columns to reflect asset-specific names (e.g., "a2" becomes "A1 GM 1 GB IP DE_a2")
        data_for_asset.columns = [f"{asset_name}_{col}" for col in required_column_names]#.................................................changes



        # Concatenate the data for this asset horizontally to the output DataFrame
        output_df = pd.concat([output_df, data_for_asset], axis=1)

    # Generate Date, Time, and Sr No columns at 30-minute intervals
    date_list = [(start_time + timedelta(minutes=30 * i)).strftime('%d %b %Y') for i in range(49)]
    time_list = [(start_time + timedelta(minutes=30 * i)).strftime('%I:%M %p') for i in range(49)]
    sr_no_list = list(range(1, 50))

    


    # Insert Date, Time, and Sr No columns into the final output DataFrame
    output_df.insert(0, 'Date', date_list)
    output_df.insert(1, 'Time', time_list)
    output_df.insert(2, 'Sr No', sr_no_list)

    # Add an empty 'Code' column at the end
    output_df['Code'] = 0

    
    # Fill NaN values in the DataFrame with 0
    output_df = output_df.fillna(0)


    # Save the processed data using ExcelWriter
    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False)

    
    # Display success message when all files are processed
    st.info(f"Data has been processed and saved")


# Create a button to trigger the process
if st.button('Preprocess Data'):
    process_data()


##########################  Classification ###############################








##########################  LSTM Autoencoder for Anomaly Detection ###############################

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# Set random seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to train the LSTM autoencoder model
def train_lstm_autoencoder_model(training_file_path, model_folder_path):
    set_random_seed()

    # Load and preprocess training data
    df = pd.read_excel(training_file_path)
    #df = df.dropna()

    
    column_names_train = df.columns[3:-1]
    X = df[[col for col in column_names_train if not col.endswith(('_d2', '_t2'))]]

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, os.path.join(model_folder_path, "lstm_auto_scaler.pkl"))

    # Reshape to 3D for LSTM (samples, time_steps, features)
    TIME_STEPS = 30
    def create_sequences(data, time_steps=TIME_STEPS):
        seqs = []
        for i in range(len(data) - time_steps):
            seqs.append(data[i:i + time_steps])
        return np.array(seqs)
    
    X_seq = create_sequences(X_scaled)

    # Define LSTM Autoencoder model
    inputs = Input(shape=(TIME_STEPS, X_seq.shape[2]))
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)
    decoded = RepeatVector(TIME_STEPS)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(X_seq.shape[2]))(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    autoencoder.fit(X_seq, X_seq, epochs=20, batch_size=64, shuffle=True)

    # Save model
    autoencoder.save(os.path.join(model_folder_path, "lstm_auto_model.h5"))

    st.success("LSTM Autoencoder training completed and model saved!")

def predict_lstm_autoencoder(test_file_path, model_folder_path):
    set_random_seed()
    import pandas as pd
    # Load test data
    df_test = pd.read_excel(test_file_path)

    column_names = df_test.columns[3:-1]
    X_test = df_test[[col for col in column_names if not col.endswith(('_d2', '_t2'))]]

    # Load scaler and scale
    scaler = joblib.load(os.path.join(model_folder_path, "lstm_auto_scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    TIME_STEPS = 30
    def create_sequences(data, time_steps=TIME_STEPS):
        seqs = []
        for i in range(len(data) - time_steps):
            seqs.append(data[i:i + time_steps])
        return np.array(seqs)

    X_test_seq = create_sequences(X_test_scaled)

    # Load model
    model = load_model(os.path.join(model_folder_path, "lstm_auto_model.h5"))

    # Predict and calculate reconstruction error per feature
    X_pred = model.predict(X_test_seq)
    errors = np.mean(np.abs(X_pred - X_test_seq), axis=1)  # shape: (samples, features)

    # Threshold (e.g., 95th percentile of average feature-wise errors)
    overall_mae = np.mean(errors, axis=1)
    threshold = np.percentile(overall_mae, 95)

    anomaly_indices = np.where(overall_mae > threshold)[0]
    feature_anomalies = np.where(errors[anomaly_indices] > np.percentile(errors, 95), 1, 0)

    sensor_dict = {}

    for idx, feature_row in enumerate(feature_anomalies):
        for f_idx, is_anomaly in enumerate(feature_row):
            if is_anomaly:
                feature_name = column_names[f_idx]
                sensor_id = feature_name.split('_')[0]  # e.g., "Sensor3" from "Sensor3_Temperature"
                if sensor_id not in sensor_dict:
                    sensor_dict[sensor_id] = {"count": 0, "params": set()}
                sensor_dict[sensor_id]["count"] += 1
                sensor_dict[sensor_id]["params"].add(feature_name)

    # Filter only sensors with anomaly count > 30
    filtered_dict = {sensor: info for sensor, info in sensor_dict.items() if info["count"] > 30}

    if not filtered_dict:
        st.session_state["check_bd_clicked"] = False
        st.success("✅ No abnormalities detected in any sensor.")
        return "✅ No abnormalities detected in any sensor."

    # Display in table format
    import pandas as pd
    table_data = {
        "Sensor Having Abnormality": [],
        "Affected Parameters": [],
        "Anomaly Count": []
    }

    for sensor, info in filtered_dict.items():
        table_data["Sensor Having Abnormality"].append(sensor)
        table_data["Affected Parameters"].append(", ".join(info["params"]))
        table_data["Anomaly Count"].append(info["count"])

    df_result = pd.DataFrame(table_data)

    st.session_state["check_bd_clicked"] = True
    st.warning("🚨 Abnormalities Detected")
    st.dataframe(df_result, use_container_width=True)

    return df_result

# Streamlit app UI
st.title("Anamoly Detector")


# Inside Streamlit UI
if st.button("Check abnormality in sensors"):
    with st.spinner("🔍 Checking for abnormality..."):
        #train_lstm_autoencoder_model(training_file_path, model_folder_path)
        result = predict_lstm_autoencoder(test_file_path, model_folder_path)
        st.session_state["Anamoly_output"] = result

        if isinstance(result, str):  # Means no anomaly case
            st.markdown(f"```text\n{result}\n```")
        else:
            st.success("✅ Anomaly detection complete!")









#..........................................Trend..............................






import matplotlib.pyplot as plt

# Mapping for parameters to descriptive names
parameter_mapping = {
    'a2': 'Acceleration',#'a2', 'vv2', 'av2', 'hv2', 't2', 'd2
    'av2': 'Axial Velocity',
    'vv2': 'Vertical Velocity',
    'hv2': 'Horizontal Velocity',
    't2': 'Temperature',
    
}

# Column types with "All" option for UI
column_types_ui = ['All'] + list(parameter_mapping.values())

# Reverse mapping for internal logic
reverse_parameter_mapping = {v: k for k, v in parameter_mapping.items()}

# Streamlit UI
st.title("Trend Visualization for Sensor Data")

# Validate files
if not os.path.exists(test_file_path) or not os.path.exists(threshold_file_path):
    st.error("Required files not found! Ensure the test and threshold file paths are correct.")
else:
    try:
        # Load test and threshold data
        test_df = pd.read_excel(test_file_path)
        threshold_df = pd.read_excel(threshold_file_path)

        if test_df.empty:
            st.warning("NO DATA in the test file.")
        else:
            # Extract alternate sensor names
            sensor_mapping = threshold_df[['Asset name', 'Sensor name']].drop_duplicates()
            asset_to_sensor = dict(zip(sensor_mapping['Asset name'], sensor_mapping['Sensor name']))

            # UI filter with alternate names
            sensor_names = list(asset_to_sensor.values())
            selected_sensor_name = st.selectbox("Select the sensor", sensor_names, index=0)

            # Map selected sensor name back to the asset name
            selected_asset = next(asset for asset, sensor in asset_to_sensor.items() if sensor == selected_sensor_name)

            selected_column_ui = st.selectbox("Select parameter", column_types_ui, index=0)

            # Map the selected UI parameter back to its internal name
            if selected_column_ui == 'All':
                selected_column = 'All'
            else:
                selected_column = reverse_parameter_mapping[selected_column_ui]

            # Check if test data contains the required columns
            if selected_column == 'All':
                asset_columns = [f"{selected_asset}_{param}" for param in parameter_mapping.keys()]
            else:
                asset_columns = [f"{selected_asset}_{selected_column}"]

            if not all(col in test_df.columns for col in asset_columns):
                st.warning("Selected asset or columns not found in the test dataset.")
            else:
                # Extract relevant data for the selected asset and column type(s)
                time_data = test_df['Time']
                date_data = test_df['Date']
                datetime_data = pd.to_datetime(date_data + ' ' + time_data, format='%d %b %Y %I:%M %p')

                # Determine start and end dates for the X-axis label
                start_date = datetime_data.min().strftime('%d %b %Y')
                end_date = datetime_data.max().strftime('%d %b %Y')

                # Generate hourly tick labels
                hourly_ticks = pd.date_range(start=datetime_data.min(), end=datetime_data.max(), freq='H')

                # Prepare the plot
                plt.figure(figsize=(12, 6))

                if selected_column == 'All':
                    # Plot all parameters for the selected asset
                    for param, display_name in parameter_mapping.items():
                        column_name = f"{selected_asset}_{param}"
                        column_data = test_df[column_name]
                        plt.plot(datetime_data, column_data, linestyle='-', label=display_name)
                else:
                    # Plot the specific parameter
                    column_data = test_df[f"{selected_asset}_{selected_column}"]
                    plt.plot(datetime_data, column_data, linestyle='-', label=selected_column_ui)

                    # Get threshold values for the selected asset and parameter
                    threshold_row = threshold_df[
                        (threshold_df['Asset name'] == selected_asset) &
                        (threshold_df['Parameter'] == selected_column)
                    ]
                    if not threshold_row.empty:
                        caution_value = threshold_row['Caution'].values[0]
                        warning_value = threshold_row['Warning'].values[0]

                        # Add horizontal lines for caution and warning thresholds
                        plt.axhline(y=caution_value, color='orange', linestyle='--', label="Caution Threshold")
                        plt.axhline(y=warning_value, color='red', linestyle='--', label="Warning Threshold")

                # Configure the plot
                plt.xlabel(f"Time ({start_date} - {end_date})")
                plt.ylabel("Values")
                plt.title(f"Trend for {selected_sensor_name} - {selected_column_ui}")
                plt.xticks(hourly_ticks, [tick.strftime('%I %p') for tick in hourly_ticks], rotation=45)
                plt.grid(True)
                plt.legend(loc='upper left')  # Place the legend in the top-left corner
                plt.tight_layout()

                # Display the plot
                st.pyplot(plt)

                # Add functionality for threshold crossing counts
                warning_counts = {}
                caution_counts = {}

                for param, display_name in parameter_mapping.items():
                    column_name = f"{selected_asset}_{param}"
                    threshold_row = threshold_df[
                        (threshold_df['Asset name'] == selected_asset) &
                        (threshold_df['Parameter'] == param)
                    ]

                    if not threshold_row.empty:
                        caution_value = threshold_row['Caution'].values[0]
                        warning_value = threshold_row['Warning'].values[0]

                        # Count how many times the parameter crosses caution and warning thresholds
                        caution_counts[display_name] = (test_df[column_name] > caution_value).sum()
                        warning_counts[display_name] = (test_df[column_name] > warning_value).sum()
                    else:
                        caution_counts[display_name] = 0
                        warning_counts[display_name] = 0

                
                
                # Combine threshold crossing counts into a single table
                combined_df = pd.DataFrame(
                    {
                        "Parameter": list(parameter_mapping.values()),
                        "Caution Crossings": [caution_counts[display_name] for display_name in parameter_mapping.values()],
                        "Warning Crossings": [warning_counts[display_name] for display_name in parameter_mapping.values()]
                    }
                )
                
               # Create a new table with Sensor Name displayed only once
                sensor_row = pd.DataFrame({"Parameter": ["Sensor Name"], "Caution Crossings": [selected_sensor_name], "Warning Crossings": [""]})
                combined_df = pd.concat([sensor_row, combined_df], ignore_index=True)

                # Adjust the column names
                combined_df.columns = ["Parameter", "Caution Crossings", "Warning Crossings"]

                # Display the combined table
                st.markdown("### Threshold Crossing frequency")
                st.table(combined_df.T)

    except Exception as e:
        st.error(f"Error processing the files: {e}")



















##def predict_lstm_autoencoder(test_file_path, model_folder_path):
##    set_random_seed()
##
##    # Load test data
##    df_test = pd.read_excel(test_file_path)
##    
##
##    column_names = df_test.columns[3:-1]
##    X_test = df_test[[col for col in column_names if not col.endswith(('_d2', '_t2'))]]
##
##    # Load scaler and scale
##    scaler = joblib.load(os.path.join(model_folder_path, "lstm_auto_scaler.pkl"))
##    X_test_scaled = scaler.transform(X_test)
##
##    # Create sequences
##    TIME_STEPS = 10
##    def create_sequences(data, time_steps=TIME_STEPS):
##        seqs = []
##        for i in range(len(data) - time_steps):
##            seqs.append(data[i:i + time_steps])
##        return np.array(seqs)
##
##    X_test_seq = create_sequences(X_test_scaled)
##
##    # Load model
##    model = load_model(os.path.join(model_folder_path, "lstm_auto_model.h5"))
##
##    # Predict and calculate reconstruction error per feature
##    X_pred = model.predict(X_test_seq)
##    errors = np.mean(np.abs(X_pred - X_test_seq), axis=1)  # shape: (samples, features)
##
##    # Threshold (e.g., 95th percentile of average feature-wise errors)
##    overall_mae = np.mean(errors, axis=1)
##    threshold = np.percentile(overall_mae, 95)
##
##    anomaly_indices = np.where(overall_mae > threshold)[0]
##    feature_anomalies = np.where(errors[anomaly_indices] > np.percentile(errors, 95), 1, 0)
##
##    sensor_dict = {}
##
##    for idx, feature_row in enumerate(feature_anomalies):
##        for f_idx, is_anomaly in enumerate(feature_row):
##            if is_anomaly:
##                feature_name = column_names[f_idx]
##                sensor_id = feature_name.split('_')[0]  # e.g., "Sensor3" from "Sensor3_Temperature"
##                if sensor_id not in sensor_dict:
##                    sensor_dict[sensor_id] = {"count": 0, "params": set()}
##                sensor_dict[sensor_id]["count"] += 1
##                sensor_dict[sensor_id]["params"].add(feature_name)
##
##    if not sensor_dict:
##        st.session_state["check_bd_clicked"] = False
##        return "✅ No abnormalities detected."
##
##    result_str = ""
##    for sensor, info in sensor_dict.items():
##        result_str += f"{sensor} is having abnormality.\n"
##        result_str += f"   ⏳ Anomaly Count: {info['count']}\n"
##        result_str += f"   🚨 Affected Parameters: {', '.join(info['params'])}\n"
##
##    st.session_state["check_bd_clicked"] = True
##    return result_str
##
##
##if st.button("Check abnormality in sensors"):
##    with st.spinner("Checking for abnormality..."):
##        #train_lstm_autoencoder_model(training_file_path, model_folder_path)
##        result = predict_lstm_autoencoder(test_file_path, model_folder_path)
##        st.session_state5["Anamoly_output"] = result
##        st.markdown(f"```text\n{result}\n```")
##        st.success("Anomaly detection complete!"
