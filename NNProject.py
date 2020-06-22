import os
import myDSP
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf


n_features = 16
trainDir = 'train'


def retInput(f1): # return the energy in the given bands
    fs,y=myDSP.readWav(f1)
    Y=abs(np.fft.fft(y))
    Y=Y[0:int(len(Y)/2)]
    
    N=len(Y)
    
    bands=np.linspace(0, fs/2, n_features + 1) # in Hertz
    energy=np.zeros(n_features)
    bandLimits=(bands*N/(fs/2)).astype('int') # in samples
    # 16,000 de esantioane si doar 8,000 sunt bune
    for index in range(n_features):
        energy[index]=np.sum(Y[bandLimits[index]:bandLimits[index+1]]) # val absoluta in transf Fourier
    return(energy)
    
X = np.zeros(n_features)

#start
classes=os.listdir(trainDir)    # reading the dataset from the train directory

T = np.zeros(len(classes))

for i in range(len(classes)):
    files = os.listdir(os.path.join(trainDir, classes[i]))  # read the recordings from the subdirectories using the list of classes
    for j in range(len(files)):
        print(os.path.join(trainDir, classes[i], files[j]))
        energy = retInput(os.path.join(trainDir, classes[i], files[j]))
        print(energy)
        X = np.vstack((X, energy)) # fiecare linie are 16 col care repr energia din benzi
                                   # In the matrix X we store the energy, so it has sixteen columns that represent the energy from the bends.
        newT = np.zeros(len(classes))
        newT[i] = 1
        T = np.vstack((T, newT))    # the 160 lines represent the value 1 allocated for each recording depending on the class it belongs to 

plt.bar(np.arange(n_features),energy)

X = np.delete(X,0,0)
T = np.delete(T,0,0)
 
n_classes=len(classes)

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)


model = Sequential()
model.add(Dense(10, input_dim=n_features, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))

# compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the Keras model on the dataset
net = model.fit(X_train, T_train, epochs=300)
# evaluate the Keras model
_,accuracy = model.evaluate(X_test, T_test)
print('Accuracy: %.2f' % (accuracy*100),'%')

# Here we record our voice:
plt.close('all')
print('Now it is time to read the words:')
samplerate = 16000  
duration = 1 # seconds
myFile = 'wow.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
sf.write(myFile, mydata, samplerate)


os.listdir('E:/Anul3 (2019-2020)/Project')
filepath='E:/Anul3 (2019-2020)/Project'
fs2,x=myDSP.readWav(inreg)

# Here we plot the recording in frequency and time:

myDSP.plotInTime(x, fs2)
myDSP.plotInFrequency(x, fs2)

print('The classes are:')
print(classes)

print('The energy for the recording is:')
inreg = 'house.wav'
myFileEnergy = retInput(inreg)
print(myFileEnergy)
myFileEnergy2 = np.zeros(len(classes))
myFileEnergy = np.vstack((myFileEnergy, myFileEnergy2)) 
myFileEnergy = np.delete(myFileEnergy,1,0)

Y_test = model.predict(myFileEnergy)
Y_test1=np.argmax(Y_test,axis=1)

print('Argmax value is:', Y_test1, 'and the word is', classes[Y_test1[0]])
