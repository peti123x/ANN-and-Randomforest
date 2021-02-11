class RandomForest():
    def __init__(self, trees, min_samples, train, test):
        #Random Forest should be taking in the same training and test data (same shuffle and length)
        #so we can initialise it with data taken from ANN
        self.trees = trees
        self.min_samples_leaf = min_samples
        self.train = train
        self.test = test
        
    def train_model(self):
        self.tree = ensemble.RandomForestClassifier(n_estimators = self.trees, criterion='entropy', min_samples_leaf=self.min_samples_leaf)
        self.tree.fit(self.train[0], self.train[1])
        
    def predict(self):
        #Predict, then measure accuracy
        predicted = self.tree.predict(self.test[0])
        accuracy = self.measure_accuracy(predicted)
        print("Prediction accuracy:" + str(accuracy))
        return accuracy
        
    def measure_accuracy(self, predicted):
        corr = 0
        for i in range(len(predicted)):
            if(int(predicted[i]) == int(test[1][i])):
                corr = corr + 1
        return float(corr/len(predicted))
    
    #This takes in an array of trees, and min samples at the leaf, trains a model each with these parameters
    #and graphs accuracies for parameters
    def measure_all_params(self, trees, min_at_leaf):
        accuracies = {}
        #For each minimum sample, train with array of trees one by one
        for minim in min_at_leaf:
            self.min_samples_leaf = minim
            accuracies[minim] = []
            for num in trees:
                #Set trees, train, then predict and save accuracy
                self.trees = num
                self.train_model()
                accuracy = self.predict()
                accuracies[minim].append(accuracy)
        #Plot
        self.plot_accuracies(accuracies, trees)
        
    def plot_accuracies(self, accuracies, trees):
        #Subplot
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        keyindex = 0
        #Colours for each number of tree
        cols = ["#000000", "#ff0000", "#ff00ff", "#5ac18e", "#0000ff"]
        for key in accuracies.keys():
            numbers = np.arange(1, len(trees)+1)
            for i in range(len(trees)):
                axs[keyindex].scatter(numbers[i], accuracies[key][i], label=str(trees[i])+" trees", color=cols[i])
            axs[keyindex].set_xlabel("Point")
            axs[keyindex].set_ylabel("Accuracy")
            axs[keyindex].set_title("Accuracy against number of trees at " + str(key) + " samples at leaf")
            axs[keyindex].grid(True)
            axs[keyindex].legend()
            keyindex = keyindex + 1
            
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
    
    def nfold_cv(self, n):
        parts = self.partition(n)
        testPartition = 0
        accuracies = []
        while testPartition < n:
            #Sets
            train = self.makeTrainSet(parts, testPartition)
            train = train.transpose()
            train_x = train[1:,].transpose()
            train_y = train[0]
            
            test = parts[testPartition].transpose()
            test_x = test[1:,].transpose()
            test_y = test[0]
            #Train
            self.tree = ensemble.RandomForestClassifier(n_estimators = self.trees, criterion='entropy', min_samples_leaf=self.min_samples_leaf)
            self.tree.fit(train_x, train_y)
            #Predict
            predicted = self.tree.predict(test_x)
            corr = 0
            for i in range(len(predicted)):
                if(int(predicted[i]) == int(test_y[i])):
                    corr = corr + 1
            accuracy = float(corr/len(predicted))
            
            print("CV on test set at index " + str(testPartition) + " has " + str(accuracy) + " accuracy")
            accuracies.append(accuracy)
            
            testPartition = testPartition + 1
        return float(sum(accuracies)/len(accuracies))
            
    def assess_params(self, data, treearr, min_samples, n):
        self.data = data
        self.min_samples_leaf = min_samples
        means = []
        for elem in treearr:
            self.trees = elem
            print("Performing " + str(n) + "-fold CV with " + str(self.trees) + " trees...")
            print("Training ...")
            mean = self.nfold_cv(n)
            print("Training and evaluation done.")
            means.append(mean)
        print("Assessment finished.")
        for i in range(len(treearr)):
            print("Mean accuracy: " + str(means[i]) + " for trees = " + str(treearr[i]))
        self.plot_params(means, treearr)
        
    def plot_params(self, means, treearr):
        cols = ["#000000", "#d35400", "#a569bd"]
        for i in range(len(means)):
            plt.scatter(i, means[i], label=str(treearr[i]) + " trees", color=cols[i])
        plt.xlabel("Tree")
        plt.ylabel("Mean accuracy")
        plt.title("10-fold CV on different number of trees")
        plt.legend()
        plt.show();
		
net_data = net.data

forest = RandomForest(1000, 5, train, test)
#Section 3
forest.measure_all_params([10, 50, 100, 1000, 5000], [5, 50])
#Section 4
forest.assess_params(net_data, [20, 500, 10000], 5, 10)