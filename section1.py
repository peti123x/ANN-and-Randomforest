import numpy as np
from matplotlib import pyplot as plt 
import pandas
import seaborn as sns

#Read coordinate csv. Col 1 is x coords and Col2 is y coords. 
def read_coords(filename):     
    myFile = open(filename) 
    row =0 
    coords =[] 
    for line in myFile:
        #skip first line as it contains labels
        if row > 0:
            coords.append(line.rstrip().split(",")[:])
        row = row+1
        #coords[row] = line.rstrip().split(",")[:] 
    myFile.close()
    return coords

def normalise(data):
    #Transpose, take out labels and convert remaining data to float
    data = data.transpose()
    label = data[0]
    data = data[1:]
    data = data.astype(float)
    #Enumerate through each col then each point in all
    #Do z = (xi - min(x))/(max(x)-min(x)) to normalise where x marks the set of numbers
    for j, col in enumerate(data):
        #Get min and max to be able to normalise
        cMax = np.amax(col)
        cMin = np.amin(col)
        for i, x in enumerate(col):
            norm = (x - cMin)/(cMax - cMin)
            #Write back
            data[j][i] = norm
    #Construct mutated data into original array with labels that we do by
    #creating new list, adding labels and the dataset, converting to np array then transposing back to original dim
    newData = []
    newData.append(label)
    newData[1:] = data
    newData = np.asarray(newData)
    newData = newData.transpose()
    return newData

def summarise(data):
    print("This set has a dimension of " , data.shape)
    #This will contain a series of tuples that charectarise each row
    summary = []
    cols = ['Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3', 'Power_range_sensor_4', 'Pressure_sensor_1', 'Pressure_sensor_2','Pressure_sensor_3', 'Pressure_sensor_4', 'Vibration_sensor_1', 'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4']
    rows = ["Mean", "Standard Deviation", "Min", "Max"]
    #Turn around so we can process numbers as rows and exclude Status
    data = data.transpose()
    data = data[1:]
    data = data.astype(float)
    for col in data:
        #Create tuple for each property (e.g. Power_range_sensor_1)
        temp = (np.mean(col), np.std(col), np.amin(col), np.amax(col))
        summary.append(temp)
    #Create table
    dataFrame = pandas.DataFrame(summary, columns=rows, index=cols)
    print(dataFrame)
    #Normally I would use Mean/Std/Min/Max as y axis labels but in this case the table would be too wide
    #So .transpose() would make it more difficult to read
    
def genBoxPlot(data):
    #Get the desired feature against the state, then zip into tuples
    joined = list(zip(data[:,0], data[:,9]))
    #Convert to np for np functionality
    joined = np.asarray(joined)
    #Find indices where the states are either normal, or abnormal and seperate them into different arrays so that we can do
    #different subplots broken down by state
    index = np.where(joined[0:,] == "Normal")[0]
    normals = joined[index][:,1]
    index = np.where(joined[0:,] == "Abnormal")[0]
    abnormals = joined[index][:,1]
    #
    normals = normals.astype(float)
    abnormals = abnormals.astype(float)
    #Set plot properties
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Normal and Abnormal state against Vibration Sensor 1')
    ax.set_ylabel("Vibration Sensor 1")
    ax.set_xlabel("State")
    #ax.set_xticklabels(["Normal", "Abnormal"])
    ax.boxplot([normals, abnormals], labels=["Normal", "Abnormal"])
    plt.show()
    
def genDensityPlot(data):
    #Get the desired feature against the state, then zip into tuples
    joined = list(zip(data[:,0], data[:,10]))
    #Convert to np for np functionality
    joined = np.asarray(joined)
    #Find indices where the states are either normal, or abnormal and seperate them into different arrays so that we can do
    #different subplots broken down by state
    index = np.where(joined[0:,] == "Normal")[0]
    normals = joined[index][:,1]
    index = np.where(joined[0:,] == "Abnormal")[0]
    abnormals = joined[index][:,1]
    normals = normals.astype(float)
    abnormals = abnormals.astype(float)

    sns.kdeplot(normals, color="green", shade=True, legend=True, label="Normal")
    sns.kdeplot(abnormals, color="red", shade=True, legend=True, label="Abnormal")
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("Vibration Sensor 2")
    plt.title("Density of values measured by Vibration Sensor 2 by state")
    plt.show()
    

data = read_coords("ML2_dataset.csv")
data = np.asarray(data) 
summarise(data)
#Normalise dataset between 0 and 1
normalised = normalise(data)
#summarise(normalised)
genBoxPlot(normalised)
genDensityPlot(normalised)