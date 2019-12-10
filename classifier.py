import numpy as np
import scipy.io as sio
from os import listdir
from os.path import join
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

def load_data(path: str=None) -> tuple:
    """Loads analysis data from ./data/"""
    dribble_path = join('data', 'test', 'dribble')
    inout_path = join('data', 'test', 'inout')

    # One-hot encoding. We transform output classes into binary values with dribble=1, and inout=0
    x, y = [], []

    # Load dribble data
    for f in listdir(dribble_path):
        s = sio.loadmat(join(dribble_path, f))['s'][0,0]
        x_freq = s['x_freq'][0]
        x_amp = s['x_amp'][0]
        z_freq = s['z_freq'][0]
        z_amp = s['z_amp'][0]
        
        x.append(np.concatenate((x_freq, x_amp, z_freq, z_amp)))
        y.append(1)
    
    # Load inout data
    for f in listdir(inout_path):
        s = sio.loadmat(join(inout_path, f))['s'][0,0]
        x_freq = s['x_freq'][0]
        x_amp = s['x_amp'][0]
        z_freq = s['z_freq'][0]
        z_amp = s['z_amp'][0]

        x.append(np.concatenate((x_freq, x_amp, z_freq, z_amp)))
        y.append(0)

    dribble_path = join('data', 'train', 'dribble')
    inout_path = join('data', 'train', 'inout')

    # Load dribble data
    for f in listdir(dribble_path):
        s = sio.loadmat(join(dribble_path, f))['s'][0,0]
        x_freq = s['x_freq'][0]
        x_amp = s['x_amp'][0]
        z_freq = s['z_freq'][0]
        z_amp = s['z_amp'][0]
        
        x.append(np.concatenate((x_freq, x_amp, z_freq, z_amp)))
        y.append(1)
    
    # Load inout data
    for f in listdir(inout_path):
        s = sio.loadmat(join(inout_path, f))['s'][0,0]
        x_freq = s['x_freq'][0]
        x_amp = s['x_amp'][0]
        z_freq = s['z_freq'][0]
        z_amp = s['z_amp'][0]

        x.append(np.concatenate((x_freq, x_amp, z_freq, z_amp)))
        y.append(0)

    return x, y

if __name__ == "__main__":
    X_train, y_train = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=1) 

    gnb = GaussianNB()
    gnb.fit(X_train, y_train) 
    y_pred = gnb.predict(X_test) 
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

