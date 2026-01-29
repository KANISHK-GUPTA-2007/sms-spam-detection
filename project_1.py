import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


#importing libraries
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split



#loading datasets
df = pd.read_csv(
    "C:/Users/LENOVO/OneDrive/Desktop/vs code coding prac/spam.csv",
    encoding="latin-1"
)
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'].to_numpy(dtype=object),
    df['label'].to_numpy(dtype=int),
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
validation_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#fetch pretrained model from tensorflow hub

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"


#checking whether we have gpu available for the training dataset
# print("GPU is", 'available' if tf.config.list_physical_devices('GPU') else 'not available')

hub_layer = hub.KerasLayer(embedding, input_shape = [], dtype = tf.string, trainable=True)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

#model summary
model.compile(optimizer = 'adam',loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(100),epochs = 25,validation_data = validation_data.batch(100),verbose = 1)

#prediction on new data
text = ["Congratulations! You won a free prize"]

prediction = model.predict(text)
probability = tf.sigmoid(prediction).numpy()[0][0]

print("Spam probability:", probability)

#model evaluation
results = model.evaluate(test_data.batch(100),verbose = 2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))