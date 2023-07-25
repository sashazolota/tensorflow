import tensorflow as tf

#DATASET
#dataset from API
fmnist = tf.keras.datasets.fashion_mnist
#loading the dataset
(x_train, y_train),(x_test, y_test) = fmnist.load_data()
#normalizing pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

#CALLBACK
#creating callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        '''
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
        '''

    # Check the loss
        if(logs.get('loss') < 0.4):
            # Stop if threshold is met
            print("\nLoss is lower than 0.4 so cancelling training!")
            self.model.stop_training = True

    # modified code to accuracy metrix
        if (logs.get('accuracy') > 0.6):
            print("\nAccuracy is more than 60%, training stops.")
            self.model.stop_training = True

    # Instantiate class
callbacks = myCallback()


#MODEL
#defining model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compiling the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_cateogrical_crossentropy',
              metrics=['accuracy'])
#training
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
