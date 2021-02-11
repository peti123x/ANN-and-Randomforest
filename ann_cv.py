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
        plt.show();
    
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
    
    #Partition data into n folds
    def partition(self, chunks):
        temp_data = self.data
        temp_data[temp_data=="Normal"] = 0
        temp_data[temp_data=="Abnormal"] = 1
        partitions = np.array_split(temp_data, 10)
        return partitions
    
    def makeTrainSet(self, parts, currTestIdx):
        #Makes train set from all partitions that arent the delegated test set
        concat = []
        for i in range(0, len(parts)):
            if i != currTestIdx:
                for row in parts[i]:
                    concat.append(row)
        concat = np.asarray(concat)
        return concat
    
    def nfold_cv(self, chunks, ep_each):
        parts = self.partition(chunks)
        testPartition = 0
        #Mean accuracy
        accuracies = []
        #Lets assign first set as test first, then use other 9 as train
        while testPartition < chunks:
            #Build new model for each iteration
            self.model = Sequential()
            self.model.add(Dense(units=self.neurons, input_shape=(12,)))
            for j in range(self.hidden):
                self.model.add(Dense(units=self.neurons, activation=self.hidden_activation, input_dim=(996,)))
            self.model.add(Dense(units=1, activation=self.end_activation))
            self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            #Then get the train set excluding the delegated test set
            train = self.makeTrainSet(parts, testPartition)
            train = train.transpose()
            train_x = train[1:,].transpose()
            train_y = train[0]
            self.history = self.model.fit(train_x, train_y, epochs=ep_each, verbose=0)

            #Network is trained, now predict on assigned test partition
            test = parts[testPartition].transpose()
            test_x = test[1:,].transpose()
            test_y = test[0]
            predicted = self.model.predict(test_x)
            score = self.model.evaluate(test_x, test_y, verbose=2)
            print("CV on test set at index " + str(testPartition) + " has " + str(score[1]) + " accuracy")
            accuracies.append(score[1])
            #Assign next partition as test
            testPartition = testPartition + 1
        mean = sum(accuracies)/len(accuracies)
        print("Mean accuracy is " + str(mean))
        return mean
        
    def assess_params(self, neuronarr, n, ep_each):
        means = []
        for elem in neuronarr:
            self.neurons = elem
            print("Performing " + str(n) + "-fold CV with " + str(self.neurons) + " neurons...")
            print("Training ...")
            mean = self.nfold_cv(n, ep_each)
            means.append(mean)
            print("Training and evaluating done.")
        print("Assessment finished.")
        for i in range(len(neuronarr)):
            print("Mean accuracy: " + str(means[i]) + " for neurons = " + str(neuronarr[i]))
        self.plot_params(means, neuronarr)
            
    def plot_params(self, means, neuronarr):
        cols = ["#000000", "#d35400", "#a569bd"]
        for i in range(len(means)):
            plt.scatter(i, means[i], label=str(neuronarr[i]) + " neurons", color=cols[i])
        plt.xlabel("Neuron")
        plt.ylabel("Mean accuracy")
        plt.title("10-fold CV on different neurons")
        plt.legend()
        plt.show();
		
net = NeuralNetwork(normalised, 0.9,500,2,'tanh','sigmoid')
net.train(1)
net.predict()
#Section 4:
net.assess_params([50, 500, 1000], 10, 150)