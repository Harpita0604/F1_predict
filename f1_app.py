import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("🏎️ F1 Finishing Position Predictor (Custom Race Input)")

# -----------------------------
# 1️⃣ Load Transformed CSV
# -----------------------------
file_path = st.text_input(
    "Enter path to transformed CSV:",
    r"C:/Users/Harpita Bakshi/Desktop/formula dataset/F1_Transformed.csv"
)

df = None  # initialize df variable

if file_path:
    try:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
        st.success("✅ CSV Loaded Successfully")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()  # Stop script if CSV can't be loaded

# -----------------------------
# 2️⃣ Only proceed if df is loaded
# -----------------------------
if df is not None:
    # -----------------------------
    # Convert relevant numeric columns
    # -----------------------------
    numeric_cols = ['grid', 'laps', 'best_qualifying_time', 'avg_lap_time', 'positionOrder']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure target exists
    if 'positionOrder' not in df.columns:
        st.error("❌ CSV must contain 'positionOrder' column as target")
        st.stop()

    y = df['positionOrder']
    X = df.drop(columns=['positionOrder'])

    # -----------------------------
    # Ensure all features numeric & fill NaNs
    # -----------------------------
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # Model Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)
    y_pred_rounded = y_pred.round().astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    exact_match = (y_pred_rounded == y_test).sum()
    accuracy_exact = exact_match / len(y_test) * 100
    top1_match = ((y_pred_rounded - y_test).abs() <= 1).sum()
    accuracy_top1 = top1_match / len(y_test) * 100

    st.subheader("📊 Model Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Exact Match Accuracy: {accuracy_exact:.2f}%")
    st.write(f"Top-1 Accuracy (±1 position): {accuracy_top1:.2f}%")

    # -----------------------------
    # Custom Input for Prediction
    # -----------------------------
    st.subheader("Predict Finishing Position for Custom Input")

    drivers_list = [col for col in X.columns if col.startswith('driverId_')]
    constructors_list = [col for col in X.columns if col.startswith('constructorId_')]

    grid = st.number_input("Grid Position:", min_value=1, max_value=25, value=10)
    laps = st.number_input("Laps Completed:", min_value=1, max_value=100, value=50)
    best_qualifying_time = st.number_input("Best Qualifying Time (seconds):", min_value=60.0, max_value=120.0, value=90.0)
    avg_lap_time = st.number_input("Average Lap Time (milliseconds):", min_value=60000, max_value=90000, value=80000)

    driver = st.selectbox("Select Driver:", [d.replace('driverId_','') for d in drivers_list])
    constructor = st.selectbox("Select Constructor:", [c.replace('constructorId_','') for c in constructors_list])

    if st.button("Predict Finishing Position"):
        # Prepare input row
        input_row = pd.DataFrame(columns=X.columns, index=[0])
        input_row.iloc[0] = 0  # initialize all zeros
        input_row['grid'] = grid
        input_row['laps'] = lapsimport streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("🏎️ F1 Finishing Position Predictor (Custom Race Input)")

# -----------------------------
# 1️⃣ Load Transformed CSV
# -----------------------------
file_path = st.text_input(
    "Enter path to transformed CSV:",
    r"C:/Users/Harpita Bakshi/Desktop/formula dataset/F1_Transformed.csv"
)

df = None  # initialize df variable

if file_path:
    try:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
        st.success("✅ CSV Loaded Successfully")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()  # Stop script if CSV can't be loaded

# -----------------------------
# 2️⃣ Only proceed if df is loaded
# -----------------------------
if df is not None:
    # -----------------------------
    # Convert relevant numeric columns
    # -----------------------------
    numeric_cols = ['grid', 'laps', 'best_qualifying_time', 'avg_lap_time', 'positionOrder']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure target exists
    if 'positionOrder' not in df.columns:
        st.error("❌ CSV must contain 'positionOrder' column as target")
        st.stop()

    y = df['positionOrder']
    X = df.drop(columns=['positionOrder'])

    # -----------------------------
    # Ensure all features numeric & fill NaNs
    # -----------------------------
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # Model Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)
    y_pred_rounded = y_pred.round().astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    exact_match = (y_pred_rounded == y_test).sum()
    accuracy_exact = exact_match / len(y_test) * 100
    top1_match = ((y_pred_rounded - y_test).abs() <= 1).sum()
    accuracy_top1 = top1_match / len(y_test) * 100

    st.subheader("📊 Model Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Exact Match Accuracy: {accuracy_exact:.2f}%")
    st.write(f"Top-1 Accuracy (±1 position): {accuracy_top1:.2f}%")

    # -----------------------------
    # Custom Input for Prediction
    # -----------------------------
    st.subheader("Predict Finishing Position for Custom Input")

    drivers_list = [col for col in X.columns if col.startswith('driverId_')]
    constructors_list = [col for col in X.columns if col.startswith('constructorId_')]

    grid = st.number_input("Grid Position:", min_value=1, max_value=25, value=10)
    laps = st.number_input("Laps Completed:", min_value=1, max_value=100, value=50)
    best_qualifying_time = st.number_input("Best Qualifying Time (seconds):", min_value=60.0, max_value=120.0, value=90.0)
    avg_lap_time = st.number_input("Average Lap Time (milliseconds):", min_value=60000, max_value=90000, value=80000)

    driver = st.selectbox("Select Driver:", [d.replace('driverId_','') for d in drivers_list])
    constructor = st.selectbox("Select Constructor:", [c.replace('constructorId_','') for c in constructors_list])

    if st.button("Predict Finishing Position"):
        # Prepare input row
        input_row = pd.DataFrame(columns=X.columns, index=[0])
        input_row.iloc[0] = 0  # initialize all zeros
        input_row['grid'] = grid
        input_row['laps'] = laps
        input_row['best_qualifying_time'] = best_qualifying_time
        input_row['avg_lap_time'] = avg_lap_time

        # Set driver & constructor one-hot
        driver_col = f'driverId_{driver}'
        constructor_col = f'constructorId_{constructor}'
        if driver_col in input_row.columns:
            input_row[driver_col] = 1
        if constructor_col in input_row.columns:
            input_row[constructor_col] = 1

        # Predict
        predicted_position = model.predict(input_row)[0]
        predicted_position_rounded = round(predicted_position)

        st.success(f"🏁 Predicted Finishing Position: {predicted_position_rounded}")

        input_row['best_qualifying_time'] = best_qualifying_time
        input_row['avg_lap_time'] = avg_lap_time

        # Set driver & constructor one-hot
        driver_col = f'driverId_{driver}'
        constructor_col = f'constructorId_{constructor}'
        if driver_col in input_row.columns:
            input_row[driver_col] = 1
        if constructor_col in input_row.columns:
            input_row[constructor_col] = 1

        # Predict
        predicted_position = model.predict(input_row)[0]
        predicted_position_rounded = round(predicted_position)

        st.success(f"🏁 Predicted Finishing Position: {predicted_position_rounded}")
if st.button("Predict Finishing Position"):
    # Prepare input row
    input_row = pd.DataFrame(columns=X.columns, index=[0])
    input_row.iloc[0] = 0  # initialize all zeros
    input_row['grid'] = grid
    input_row['laps'] = laps
    input_row['best_qualifying_time'] = best_qualifying_time
    input_row['avg_lap_time'] = avg_lap_time

    # Set driver & constructor one-hot
    driver_col = f'driverId_{driver}'
    constructor_col = f'constructorId_{constructor}'
    if driver_col in input_row.columns:
        input_row[driver_col] = 1
    if constructor_col in input_row.columns:
        input_row[constructor_col] = 1

    # Predict
    predicted_position = model.predict(input_row)[0]
    predicted_position_rounded = round(predicted_position)
    st.success(f"🏁 Predicted Finishing Position: {predicted_position_rounded}")

    # -----------------------------
    # Top-3 / Podium Probability
    # -----------------------------
    # Get predictions from all trees
    all_tree_preds = [tree.predict(input_row)[0] for tree in model.estimators_]
    podium_count = sum([1 for p in all_tree_preds if round(p) <= 3])
    podium_probability = podium_count / len(all_tree_preds) * 100

    st.info(f"🥉 Podium Probability (Top-3): {podium_probability:}")