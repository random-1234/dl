import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# Load and prepare data
data = pd.read_csv('sonar_dataset.csv')
X = StandardScaler().fit_transform(data.drop('Target', axis=1))
y = LabelEncoder().fit_transform(data['Target'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(60, activation='relu', input_dim=60),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate model
model.fit(X_train, y_train, epochs=50, validation_split=0.1)
print("Test Accuracy:", model.evaluate(X_test, y_test)[1])
