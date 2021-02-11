from sklearn import ensemble

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
            axs[keyindex].set_xlabel("Number of trees")
            axs[keyindex].set_ylabel("Accuracy")
            axs[keyindex].set_title("Accuracy against number of trees at " + str(key) + " samples at leaf")
            axs[keyindex].grid(True)
            axs[keyindex].legend()
            keyindex = keyindex + 1

net = NeuralNetwork(normalised, 0.9,500,2,'tanh','sigmoid')
net.train(150)
net.predict()

train = net.train
test = net.test
#Run random forest
forest = RandomForest(1000, 5, train, test)
forest.measure_all_params([10, 50, 100, 1000, 5000], [5, 50])