from keras.models import Sequential
from keras.layers import Dense

class NeuralNetwork:
    def __init__(self, data, trainProp, neurons, hiddenLayers, hiddenActiv, endActiv):
        self.data = data
        self.trainProp = trainProp
        self.hidden = hiddenLayers
        self.neurons = neurons
        self.hidden_activation = hiddenActiv
        self.end_activation = endActiv

    def splitData(self, trainProp = 0.9):
        #Shuffle
        np.random.shuffle(self.data)
        elems = len(data)
        #Define set proportions we want
        test_prop = 1 - trainProp

        #Slice data array into two seperate arrays depending on proportions
        #Train 0 to elems*proportion, test elems*proportion-1 to last
        train_data = self.data[0:int(elems*trainProp)].transpose()
        test_data = self.data[int(elems*trainProp)-1:-1].transpose()
        #Split then transform string state to booleans
        x_train = train_data[1:,].transpose()
        y_train = train_data[0]
        y_train[y_train=="Normal"] = 0
        y_train[y_train=="Abnormal"] = 1

        x_test = test_data[1:,].transpose()
        y_test = test_data[0]
        y_test[y_test=="Normal"] = 0
        y_test[y_test=="Abnormal"] = 1

        return [x_train,y_train], [x_test,y_test]
    
    def plotTrainAccuracy(self):
        #Plot accuracy against epochs
        plt.plot(self.history.history['accuracy'])
        plt.ylabel("Model accuracy")
        plt.xlabel("Epoch")
        plt.title("Model training accuracy over epochs")
        plt.show()
    
    def train(self, ep = 1):
        #Split data and save as class variables
        self.train, self.test = self.splitData(self.trainProp)
        #Init Keras, add layers
        self.model = Sequential()
        #First layer takes in 12 features and all rows
        self.model.add(Dense(units=self.neurons, input_shape=(12,)))
        #For all hidden layers, add
        for i in range(self.hidden):
            self.model.add(Dense(units=self.neurons, activation=self.hidden_activation, input_dim=(996,)))
        #Add output layer, then compile and fit, then plot at the end of training
        self.model.add(Dense(units=1, activation=self.end_activation))
        self.model.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
        self.history = self.model.fit(self.train[0], self.train[1], epochs=ep)
        self.plotTrainAccuracy()
    
    def predict(self):
        predicted = self.model.predict(self.test[0])
        #Calc accuracy, between predicted and ground y
        test_y = np.asarray(self.test[1]).transpose()
        test_y = test_y.astype(float)
        
        corr = 0 
        for i in range(len(test_y)):
            if int(np.round(predicted[i][0])) == int(test_y[i]):
                corr = corr + 1
        print("Prediction accuracy:" + str(float(corr/len(test_y))))
        
        score = self.model.evaluate(self.test[0], self.test[1], verbose=2)
        print(score) #keras built in scoring

#This is defined in previous section
normalised = normalise(data)
net = NeuralNetwork(normalised, 0.9,500,2,'tanh','sigmoid')
net.train(150)
net.predict()