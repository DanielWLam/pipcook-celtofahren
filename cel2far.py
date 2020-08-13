import numpy as np
import tensorflow as tf

def fit(num):
  celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
  fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
  L0 = tf.keras.layers.Dense(units=1) # 全连接层：每个神经元都与上层所有神经元相连接
  model = tf.keras.Sequential([L0])
  model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
  history = model.fit(celsius_q, fahrenheit_a, epochs=2000, verbose=0) # epochs: 训练次数，verbose：训练日志
  print(model.predict([num])) # 向模型输入100.0

def lang():
 
    return "python"