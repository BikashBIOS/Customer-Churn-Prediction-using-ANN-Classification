# 🧠 ANN Classification & Regression — Customer Churn & Salary Prediction

This project demonstrates how to build, train, tune, and deploy **Artificial Neural Networks (ANNs)** using TensorFlow/Keras on the **Churn_Modelling** dataset.

**Two models are built:**
- 🔴 **Classification** → Predict whether a customer will churn (Yes / No)
- 🟢 **Regression** → Predict the estimated salary of a customer

---

## 📁 Project Structure

```
├── experiments.ipynb           # ANN Classification — Data prep, model training, TensorBoard
├── prediction.ipynb            # ANN Classification — Load model & predict on new input
├── app.py                      # ANN Classification — Streamlit web app
├── regression.ipynb            # ANN Regression — Data prep, model training, evaluation
├── streamlit_regression.py     # ANN Regression — Streamlit web app
├── hyperparametertuning.ipynb  # Grid Search for best ANN architecture
├── model.h5                    # Saved classification model
├── regression_model.h5         # Saved regression model
├── label_encoder_gender.pkl    # Saved Gender label encoder
├── onehot_encoder_geo.pkl      # Saved Geography one-hot encoder
├── scaler.pkl                  # Saved StandardScaler
└── Churn_Modelling.csv         # Source dataset (10,000 bank customers)
```

---

## 📊 Dataset Overview

The dataset (`Churn_Modelling.csv`) contains 10,000 bank customer records with the following features:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France / Germany / Spain) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card? (0 / 1) |
| `IsActiveMember` | Is an active member? (0 / 1) |
| `EstimatedSalary` | Estimated salary |
| `Exited` | **Target for Classification** — Did the customer leave? (0 / 1) |

---

## ⚙️ Common Preprocessing (Used in All Files)

Before building any model, the dataset goes through the same preprocessing pipeline.

### Step 1 — Load & Clean Data

```python
data = pd.read_csv('Churn_Modelling.csv')

# Drop columns that don't help in prediction
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```

> `RowNumber`, `CustomerId`, and `Surname` are identifiers — they carry no predictive signal and are removed.

---

### Step 2 — Encode the `Gender` Column (Label Encoding)

```python
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
```

> `LabelEncoder` converts text labels to numbers — `Female → 0`, `Male → 1`. This is used instead of one-hot encoding because Gender has only 2 values.

---

### Step 3 — Encode the `Geography` Column (One-Hot Encoding)

```python
onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Merge encoded columns back into the main dataframe
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
```

> `OneHotEncoder` converts `Geography` into 3 binary columns: `Geography_France`, `Geography_Germany`, `Geography_Spain`. This avoids creating an artificial ordinal relationship between countries.

---

### Step 4 — Train/Test Split + Feature Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit on train, then transform
X_test  = scaler.transform(X_test)        # Only transform (never fit on test)
```

> `StandardScaler` normalizes features to have mean=0 and std=1. This is crucial for neural networks to converge faster and avoid one feature dominating others. The scaler is **only fitted on training data** to prevent data leakage.

---

### Step 5 — Save Encoders & Scaler

```python
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
```

> The encoders and scaler are saved as `.pkl` files so they can be reloaded during prediction without refitting — ensuring the same transformation is applied to new data.

---

---

## 🔴 ANN Classification — `experiments.ipynb`

**Goal:** Predict whether a customer will churn (`Exited = 1`) or not (`Exited = 0`).

### Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    Dense(32, activation='relu'),                                    # Hidden Layer 2
    Dense(1,  activation='sigmoid')                                  # Output Layer
])
```

| Layer | Neurons | Activation | Purpose |
|---|---|---|---|
| Hidden Layer 1 | 64 | ReLU | Extract complex patterns from 12 features |
| Hidden Layer 2 | 32 | ReLU | Further compress and refine patterns |
| Output Layer | 1 | Sigmoid | Output a probability between 0 and 1 |

> **ReLU** (Rectified Linear Unit): Outputs `max(0, x)` — fast to compute and avoids the vanishing gradient problem.
>
> **Sigmoid**: Squashes output to `[0, 1]` — perfect for binary classification (churn probability).

---

### Compile the Model

```python
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```

> **Adam** is an adaptive optimizer — it adjusts the learning rate automatically per parameter.
>
> **Binary Crossentropy** is the standard loss for binary (Yes/No) classification problems.

---

### Callbacks — TensorBoard & Early Stopping

```python
# Log training metrics for visualization in TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Stop training when validation loss stops improving
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

> **TensorBoard** saves training logs so you can visualize loss/accuracy curves.
>
> **EarlyStopping** with `patience=10` means: if validation loss doesn't improve for 10 consecutive epochs, stop training and restore the best weights — preventing overfitting.

---

### Train the Model

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[tensorflow_callback, early_stopping_callback]
)
```

> Trains on 8,000 rows and validates on 2,000 rows. Even though `epochs=100` is set, Early Stopping will halt training earlier if validation loss plateaus.

---

### Save the Model

```python
model.save('model.h5')
```

> Saves the trained model to disk in HDF5 format so it can be reloaded without retraining.

---

### Launch TensorBoard

```python
%reload_ext tensorboard
%tensorboard --logdir logs/fit/
```

> Opens an interactive dashboard inside the notebook to visualize training/validation loss and accuracy over epochs.

---

---

## 🔮 ANN Classification Prediction — `prediction.ipynb`

**Goal:** Load the saved model and encoders, then predict churn on a single new customer.

### Load Model & Artifacts

```python
model = load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
```

---

### Define New Customer Input

```python
input_data = {
    'CreditScore': 600, 'Geography': 'France', 'Gender': 'Male',
    'Age': 40, 'Tenure': 3, 'Balance': 60000,
    'NumOfProducts': 2, 'HasCrCard': 1,
    'IsActiveMember': 1, 'EstimatedSalary': 50000
}
```

---

### Preprocess & Predict

```python
# Encode Gender
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# One-hot encode Geography
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Merge and scale
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]   # → 0.032 (3.2% chance of churning)

if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
```

> The model returned ~`0.032`, meaning only a 3.2% probability of churn — the customer is **not** likely to churn.

---

---

## 🌐 Classification Web App — `app.py`

**Goal:** A Streamlit UI where users enter customer details and get a real-time churn prediction.

### Load Saved Assets

```python
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
```

### UI Inputs

```python
st.title('Customer Churn Prediction')

geography      = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender         = st.selectbox('Gender', label_encoder_gender.classes_)
age            = st.slider('Age', 18, 92)
balance        = st.number_input('Balance')
credit_score   = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure         = st.slider('Tenure', 0, 10)
num_of_products  = st.slider('Number of Products', 1, 4)
has_cr_card      = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
```

> All inputs are collected using Streamlit widgets (`selectbox`, `slider`, `number_input`).

### Build, Transform & Predict

```python
# Build a dataframe from user inputs
input_data = pd.DataFrame({ 'CreditScore': [credit_score], 'Gender': [label_encoder_gender.transform([gender])[0]], ... })

# One-hot encode Geography and merge
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale and predict
input_data_scaled = scaler.transform(input_data)
prediction_proba = model.predict(input_data_scaled)[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
```

**Run the app with:**
```bash
streamlit run app.py
```

---

---

## 🟢 ANN Regression — `regression.ipynb`

**Goal:** Predict a customer's `EstimatedSalary` (a continuous number) instead of a Yes/No class.

> All preprocessing steps (loading, cleaning, encoding, scaling, saving) are **identical** to the classification notebook. The only difference is the target variable and the model design.

### Key Difference — Target Variable

```python
X = data.drop('EstimatedSalary', axis=1)   # All columns except salary
y = data['EstimatedSalary']                 # Target is now a continuous number
```

---

### Regression Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    Dense(32, activation='relu'),                                    # Hidden Layer 2
    Dense(1)                                                         # Output Layer — NO activation!
])

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
```

> **Key difference from Classification:** The output layer has **no activation function** — it outputs any real number directly, which is what we want for salary prediction.
>
> **MAE (Mean Absolute Error)** is used as the loss — it measures the average dollar difference between predicted and actual salary.

---

### Callbacks & Training

```python
log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping_callback, tensorboard_callback]
)
```

---

### Evaluate & Save

```python
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')   # Average prediction error in salary units

model.save('regression_model.h5')
```

---

---

## 🌐 Regression Web App — `streamlit_regression.py`

**Goal:** A Streamlit UI where users enter customer details and get a predicted estimated salary.

### Key Difference from Classification App

```python
st.title('Estimated Salary Prediction')

# Extra input — Exited (whether the customer churned)
exited = st.selectbox('Exited', [0, 1])

# Input dataframe includes 'Exited' instead of asking for EstimatedSalary
input_data = pd.DataFrame({
    ...,
    'Exited': [exited]
})

# Predict salary (a number, not a probability)
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: {predicted_salary:.2f}')
```

> Unlike the classification app, the output here is a **dollar figure** — not a probability or class label.

**Run the app with:**
```bash
streamlit run streamlit_regression.py
```

---

---

## 🔧 Hyperparameter Tuning — `hyperparametertuning.ipynb`

**Goal:** Automatically find the best number of neurons and layers for the ANN using Grid Search.

### Define a Flexible Model Builder

```python
def create_model(neurons=32, layers=1):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))

    # Add additional hidden layers dynamically
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

> This function creates an ANN where both the **number of hidden layers** and **neurons per layer** are configurable — perfect for Grid Search.

---

### Wrap Model for Scikit-Learn Compatibility

```python
model = KerasClassifier(layers=1, neurons=32, build_fn=create_model, verbose=1)
```

> `KerasClassifier` (from `scikeras`) wraps the Keras model so it works with Scikit-Learn's `GridSearchCV`.

---

### Define the Search Space

```python
param_grid = {
    'neurons': [16, 32, 64, 128],   # Try 4 different neuron counts
    'layers':  [1, 2],               # Try 1 or 2 hidden layers
    'epochs':  [50, 100]             # Try 50 or 100 training epochs
}
```

> This creates **4 × 2 × 2 = 16 combinations**, each evaluated using 3-fold cross-validation → **48 total training runs**.

---

### Run Grid Search

```python
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

> `n_jobs=-1` runs all combinations in parallel using all available CPU cores.
>
> The best combination (e.g., `neurons=64, layers=2, epochs=100`) is printed at the end and should be used in `experiments.ipynb` for the final model.

---

---

## 🆚 Classification vs Regression — Quick Comparison

| Aspect | Classification (`experiments.ipynb`) | Regression (`regression.ipynb`) |
|---|---|---|
| **Target** | `Exited` (0 or 1) | `EstimatedSalary` (a number) |
| **Output Activation** | `sigmoid` | None (linear) |
| **Loss Function** | `binary_crossentropy` | `mean_absolute_error` |
| **Metric** | `accuracy` | `mae` |
| **Output Meaning** | Probability of churn | Predicted salary in dollars |
| **Saved Model** | `model.h5` | `regression_model.h5` |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy streamlit scikeras pickle5
```

### Run Classification App

```bash
streamlit run app.py
```

### Run Regression App

```bash
streamlit run streamlit_regression.py
```

### Open TensorBoard

```bash
tensorboard --logdir logs/fit/             # For classification
tensorboard --logdir regressionlogs/fit/   # For regression
```

