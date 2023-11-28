from keras.models import Sequential
from keras.layers import Dense, Flatten

class Model_Classification_Numbers:
    def __init__(self,
                optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics='tf.keras.metrics.SparseCategoricalAccuracy()'):
        self.model = Sequential([
            Flatten(input_shape=(28,28)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')])
        self.optimizer = optimizer,
        self.loss = loss,
        self.metrics = metrics,
    
    def compile_model(self):
        self.model.compile(self.optimizer, self.loss, self.metrics)

    def train(self, x, y, epochs):
        self.model.fit(x, y, epochs)

        
class Model_Classification_Binary:
    def __init__(self, 
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics='tf.keras.metrics.SparseCategoricalAccuracy()'):
        model = Sequential([
            Flatten(input_shape=(28,28)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')])
        self.optimizer = optimizer,
        self.loss = loss,
        self.metrics = metrics,
    
    def compile_model(self):
        self.model.compile(self.optimizer, self.loss, self.metrics)
        
    