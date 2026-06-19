# 🧠 ANN Classification & Regression — Customer Churn & Salary Prediction

This project demonstrates how to build, train, tune, and deploy **Artificial Neural Networks (ANNs)** using TensorFlow/Keras on the **Churn_Modelling** dataset.

**Two models are built:**
- 🔴 **Classification** → Predict whether a customer will churn (Yes / No)
- 🟢 **Regression** → Predict the estimated salary of a customer

**Major Dependencies:**
- This is built on Python 3.11 -> Make sure you already have installed Python 3.11 in your local. (Due to Tensorflow).
- Create virtual env of 3.11 python -> **py -3.11 -m venv .venv** (Make sure you have Python installer installed in your local -> To enable working of **py** command).

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

## To run the file :
Execute in terminal -> **streamlit run app.py/regression.py** according to the problem statement.


## To deploy the file : 
1. Push the code to github.
2. Open streamlit cloud website. 
3. Create a new app and choose to deploy it from github.
4. Provide the details and your .py file which you want to deploy.
5. Click on deploy -> And it will create your App in your localhost.




# 🎬 Simple RNN — IMDB Movie Review Sentiment Analysis

This project demonstrates how to build a **Recurrent Neural Network (RNN)** using TensorFlow/Keras to classify movie reviews from the IMDB dataset as **Positive** or **Negative**.

It also covers the foundational concept of **Word Embeddings** through a standalone notebook before jumping into the full RNN pipeline.

---

## 📁 Project Structure

```
├── embedding.ipynb         # Standalone demo — One-Hot Encoding & Word Embeddings concept
├── simplernn.ipynb         # Main notebook — Build & train Simple RNN on IMDB dataset
├── prediction.ipynb        # Load saved model & predict sentiment on new reviews
├── main.py                 # Streamlit web app for live sentiment prediction
└── simple_rnn_imdb.h5      # Saved trained RNN model
```

---

## 🧠 Concept Overview

| Term | What it means |
|---|---|
| **Sentiment Analysis** | Classifying text as Positive or Negative |
| **RNN (Recurrent Neural Network)** | A neural network that processes sequences word-by-word, carrying memory from previous words |
| **Embedding** | Converts each word (integer ID) into a dense vector of floats that captures meaning |
| **Padding** | Makes all reviews the same length by adding zeros at the start |
| **IMDB Dataset** | 50,000 movie reviews (25k train / 25k test), pre-encoded as integers by Keras |

---

---

## 📘 `embedding.ipynb` — Word Embedding Concept (Standalone Demo)

**Goal:** Understand how raw text sentences are converted into numerical vectors before being fed into any neural network. This notebook is a learning exercise — it is **not** connected to the IMDB classification pipeline.

---

### Step 1 — Import One-Hot Encoder

```python
from tensorflow.keras.preprocessing.text import one_hot
```

> `one_hot` from Keras hashes each word into a random integer within a fixed vocabulary range. It's a simple way to convert words to numbers.

---

### Step 2 — Define Sample Sentences

```python
sent = [
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good',
]
```

> A small list of 7 sentences used to demonstrate encoding concepts.

---

### Step 3 — Define Vocabulary Size

```python
voc_size = 10000
```

> The vocabulary size sets the range for the hash function. Each word gets mapped to a number between `1` and `10000`. Larger vocabularies reduce the chance of two different words mapping to the same number (called a **hash collision**).

---

### Step 4 — One-Hot Encode Each Sentence

```python
one_hot_repr = [one_hot(words, voc_size) for words in sent]
one_hot_repr
```

> Each word in every sentence is replaced by a random integer (its hash index). For example, `'the glass of milk'` might become `[456, 7823, 312, 9001]`. The numbers differ each run because of the hash function, but the vocabulary size stays fixed.

---

### Step 5 — Pad All Sentences to the Same Length

```python
sent_length = 8
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)
```

> Neural networks require all inputs to be the same size. `pad_sequences` with `padding='pre'` adds zeros **at the beginning** of shorter sentences to make them all length 8. For example, a 4-word sentence `[456, 7823, 312, 9001]` becomes `[0, 0, 0, 0, 456, 7823, 312, 9001]`.

---

### Step 6 — Define Embedding Dimensions & Build Embedding Model

```python
dim = 10   # Each word will be represented as a 10-dimensional vector

model = Sequential()
model.add(Embedding(voc_size, dim, input_length=sent_length))
model.compile('adam', 'mse')
model.summary()
```

> The `Embedding` layer is the core concept here:
> - `voc_size = 10000` → the number of possible unique word IDs
> - `dim = 10` → each word ID is mapped to a vector of 10 floating-point numbers
> - These 10 numbers are **learned during training** — words that appear in similar contexts get similar vectors
>
> This is far more powerful than one-hot encoding because it captures **semantic meaning**.

---

### Step 7 — Predict (View Embedding Vectors)

```python
model.predict(embedded_docs)         # View embeddings for all sentences
model.predict(embedded_docs[[0]])    # View embedding for the first sentence only
```

> Since the model isn't trained on a task yet (no labels), this just shows the **randomly initialized embedding vectors** — what each word looks like as a 10-number vector before training begins. After training on a real task, these vectors would capture actual word meaning.

---

---

## 🔁 `simplernn.ipynb` — Train Simple RNN on IMDB Dataset

**Goal:** Build an end-to-end pipeline — load IMDB data, preprocess it, build a Simple RNN, train it, and save the model.

---

### Step 1 — Import Libraries

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
```

> These imports bring in everything needed: data loading, sequence padding, and the model building blocks.

---

### Step 2 — Load the IMDB Dataset

```python
max_features = 10000   # Only keep the top 10,000 most common words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Testing data shape:  {X_test.shape},  Testing labels shape:  {y_test.shape}')
```

> `imdb.load_data()` returns reviews **already encoded as integers** — every word is replaced by its rank in the vocabulary (`1` = most common word). `num_words=10000` means any word ranked lower than 10,000 is discarded (replaced with a placeholder).
>
> Labels: `y = 1` means Positive review, `y = 0` means Negative review.

---

### Step 3 — Inspect a Sample Review

```python
sample_review = X_train[0]
sample_label  = y_train[0]

print(f"Sample review (as integers): {sample_review}")
print(f"Sample label: {sample_label}")   # 1 = Positive, 0 = Negative
```

> Shows what the raw data looks like — a list of integers, not words yet. This helps confirm the data loaded correctly.

---

### Step 4 — Decode Integers Back to Words (for Understanding)

```python
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
decoded_review
```

> `imdb.get_word_index()` returns a dictionary of `word → integer`. Reversing it lets us map `integer → word`.
>
> The `i - 3` offset exists because IMDB reserves indices 0, 1, 2 for special tokens (`<pad>`, `<start>`, `<unknown>`), so all real words are shifted by 3. Words not found in the index are replaced with `'?'`.

---

### Step 5 — Pad All Reviews to the Same Length

```python
max_len = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test  = sequence.pad_sequences(X_test,  maxlen=max_len)
```

> Reviews in IMDB have different lengths (some 80 words, some 2,000+ words). The RNN requires a fixed input length. `pad_sequences` truncates reviews longer than 500 words and pads shorter reviews with zeros at the start. 500 was chosen as a reasonable balance between capturing enough context and keeping computation manageable.

---

### Step 6 — Build the Simple RNN Model

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))   # Layer 1: Embedding
model.add(SimpleRNN(128, activation='relu'))                     # Layer 2: RNN
model.add(Dense(1, activation='sigmoid'))                        # Layer 3: Output

model.build(input_shape=(None, max_len))
model.summary()
```

**How each layer works:**

| Layer | Config | What it does |
|---|---|---|
| `Embedding` | vocab=10000, dims=128 | Converts each integer word ID into a 128-dimensional vector |
| `SimpleRNN` | units=128, ReLU | Reads the 500-step sequence word-by-word, building a hidden state that accumulates context |
| `Dense` | units=1, Sigmoid | Squashes the final RNN state into a single number between 0 and 1 (churn probability) |

> **Why SimpleRNN?** It processes input sequentially — the output at each time step feeds into the next step. This allows the network to consider word order and context, unlike a regular Dense network which treats each word independently.
>
> **Why ReLU in RNN?** Avoids vanishing gradients during training. (Note: LSTMs and GRUs are better for long sequences, but SimpleRNN is great for learning the concept.)

---

### Step 7 — Compile the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

> Same setup as binary classification:
> - **Adam** — adaptive optimizer
> - **Binary Crossentropy** — loss for Yes/No classification
> - **Accuracy** — human-readable metric to monitor

---

### Step 8 — Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

> Stops training if `val_loss` doesn't improve for 5 consecutive epochs and restores the weights from the best epoch. Prevents overfitting on the training reviews.

---

### Step 9 — Train the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
```

> - `epochs=10` — maximum 10 passes through training data (Early Stopping may end it sooner)
> - `batch_size=32` — updates weights after every 32 reviews
> - `validation_split=0.2` — holds back 20% of training data to monitor overfitting after each epoch

---

### Step 10 — Save the Model

```python
model.save('simple_rnn_imdb.h5')
```

> Saves the full trained model (architecture + weights) to disk for use in `prediction.ipynb` and `main.py`.

---

---

## 🔮 `prediction.ipynb` — Load Model & Predict on New Reviews

**Goal:** Reload the saved model and run sentiment prediction on any custom movie review text.

---

### Step 1 — Load Libraries & Word Index

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
```

> The word index must be reloaded every time to ensure our text encoding matches exactly what the model was trained on.

---

### Step 2 — Load the Saved Model

```python
model = load_model('simple_rnn_imdb.h5')
model.summary()
model.get_weights()   # Inspect the trained weight matrices
```

> Restores the complete model from disk. `model.summary()` confirms the architecture, and `model.get_weights()` lets you inspect the actual learned parameters (useful for debugging or understanding).

---

### Step 3 — Helper Function: Decode a Review

```python
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
```

> Converts a list of integers back into readable words. The `i - 3` offset accounts for the 3 reserved special tokens. Useful for verifying that encoding and decoding are working correctly.

---

### Step 4 — Helper Function: Preprocess Raw Text

```python
def preprocess_text(text):
    words = text.lower().split()                                     # Lowercase & tokenize
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Word → integer
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Pad to 500
    return padded_review
```

**Step-by-step breakdown:**

- `text.lower().split()` → converts `"Great Movie!"` to `['great', 'movie!']`
- `word_index.get(word, 2)` → looks up each word's integer ID; unknown words default to `2` (the `<unknown>` token)
- `+ 3` → applies the same offset used during IMDB dataset encoding so indices align
- `pad_sequences(..., maxlen=500)` → pads or truncates to exactly 500 tokens, matching training input shape

---

### Step 5 — Prediction Function

```python
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]
```

> - Passes the preprocessed review through the model
> - `prediction[0][0]` — extracts the single scalar output (between 0 and 1)
> - `> 0.5` → Positive, `≤ 0.5` → Negative

---

### Step 6 — Run a Prediction

```python
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."

sentiment, score = predict_sentiment(example_review)

print(f'Review:           {example_review}')
print(f'Sentiment:        {sentiment}')
print(f'Prediction Score: {score}')
```

> Runs the full pipeline on a sample review. A score close to `1.0` means strongly Positive; close to `0.0` means strongly Negative.

---

---

## 🌐 `main.py` — Streamlit Web App

**Goal:** A browser-based interface where anyone can type a movie review and instantly get a sentiment prediction — no coding required.

---

### Load Libraries & Assets

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')
```

> The same word index and model are loaded once when the app starts. Streamlit caches this so it doesn't reload on every user interaction.

---

### Helper Functions (Same as `prediction.ipynb`)

```python
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, 9999) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
```

> One small difference from `prediction.ipynb`: `min(..., 9999)` clamps word indices to stay within the vocabulary size of 10,000, preventing out-of-bounds errors on unusual words.

---

### Streamlit UI

```python
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')   # Multi-line text box for the review
```

> `st.title` and `st.write` render text. `st.text_area` creates a multi-line input box for the user to paste or type their review.

---

### Classify Button & Output

```python
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
```

> When the user clicks **Classify**:
> 1. The raw text from `user_input` is passed through `preprocess_text()`
> 2. The model predicts a score
> 3. The sentiment label and raw score are displayed on screen
>
> Before clicking, a placeholder message is shown instead.

**Run the app with:**
```bash
streamlit run main.py
```

---

---

## 🔄 End-to-End Pipeline Summary

```
Raw Text Review
      ↓
  lowercase + split into words
      ↓
  word → integer (via word_index)
      ↓
  pad/truncate to 500 tokens
      ↓
  Embedding Layer  →  128-dim vector per word
      ↓
  SimpleRNN (128 units)  →  single context vector
      ↓
  Dense (sigmoid)  →  score between 0.0 and 1.0
      ↓
  score > 0.5 → Positive | score ≤ 0.5 → Negative
```

---

## 🆚 `embedding.ipynb` vs `simplernn.ipynb` — Key Differences

| Aspect | `embedding.ipynb` | `simplernn.ipynb` |
|---|---|---|
| **Purpose** | Conceptual demo of embeddings | Full end-to-end classification |
| **Dataset** | 7 hand-written sentences | 50,000 IMDB reviews |
| **Task** | No prediction task (unsupervised) | Binary sentiment classification |
| **Layers** | Embedding only | Embedding + SimpleRNN + Dense |
| **Output** | Raw embedding vectors | Positive / Negative label |
| **Encoding method** | `one_hot()` hash function | Integer rank from IMDB word index |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow numpy streamlit
```

### Run the Notebooks in Order

```
1. embedding.ipynb        ← Understand embeddings first (optional but recommended)
2. simplernn.ipynb        ← Train the model → generates simple_rnn_imdb.h5
3. prediction.ipynb       ← Test predictions in a notebook environment
4. streamlit run main.py  ← Launch the web app
```

### Launch the Web App

```bash
streamlit run main.py
```

---

## 📌 Key Concepts Recap

| Concept | Simple Explanation |
|---|---|
| **One-Hot Encoding** | Each word gets a random integer ID within a vocabulary range |
| **Padding** | Adds zeros to make all sequences the same length |
| **Embedding Layer** | Converts integer word IDs into meaningful float vectors (learned during training) |
| **SimpleRNN** | Processes words one-by-one, carrying a "memory" (hidden state) from word to word |
| **Binary Crossentropy** | Loss function for Yes/No classification |
| **Sigmoid Output** | Maps final RNN output to a probability between 0 (Negative) and 1 (Positive) |
| **EarlyStopping** | Stops training automatically when validation performance stops improving |
