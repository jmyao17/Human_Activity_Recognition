
import pandas as pd
import numpy as np

from IPython.display import Image

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
import matplotlib.ticker as ticker
from matplotlib.mlab import griddata
from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


from sklearn.svm import LinearSVC, SVC 
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, f1_score

import xgboost


import time
import keras 
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
 
#import tensorflow 
from keras.models import Sequential
from keras import backend as K
#from tensorflow.keras import backend as K
#import keras_metrics as km

from ipywidgets import IntProgress
from IPython.display import display
import time 

import statsmodels.api as sm

from scipy.stats import skew,kurtosis
from scipy.stats import norm

 import hickle as hkl


import warnings
warnings.filterwarnings('ignore')



def load_data(features):
    dfs = []
    for i in range(1,16):
        
        filename = 'Activity Recognition from Single Chest-Mounted Accelerometer/'+str(i)+'.csv'
        
        df_temp = pd.read_csv(filename)
        df_temp.columns = [features]
        dfs.append(df_temp)
        
        print("Time Duration for the participant {} is {} seconds".format(i,np.ceil(df_temp.shape[0]/52)))
    return dfs

 


def set_figstyle(xmin,xmax,ymin,ymax,xlabel,ylabel,how):
    """
        how --  True: label both sides of y;
        False: only label the left side
        """
    plt.xlabel(xlabel,fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    plt.tick_params(which='major',direction='in',length=10,labelsize=20)
    plt.tick_params(which='minor',direction='in',length=5,labelsize=20)
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=how,
                    bottom=True, top=False, left=True, right=True)

def plot_score4each(cm):
    df = print_score4each(cm)
    
    plt.figure(figsize=(10,6))
    
    plt.errorbar(df['Label_p'],df['precision'],yerr=(df['precision_Up']-df['precision_Low'])/2, fmt='o',color='r',label='precision')
    plt.errorbar(df['Label_p'],df['recall'],yerr=(df['recall_Up']-df['recall_Low'])/2, fmt='s',color='b',label='recall')
    
    plt.xlabel('Class',fontsize=24)
    plt.ylabel('Metrics',fontsize=24)
    plt.tick_params(direction='in', length=8, width=2)
    plt.legend(fontsize=24)
    plt.ylim(0, 1);
    
    plt.tick_params(which='major',direction='in',length=10,labelsize=20)
    plt.tick_params(which='minor',direction='in',length=5,labelsize=20)
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                    bottom=True, top=False, left=True, right=True)

def plot_data4AllClass(df,features):
    
    x1 = features[0]
    ys = features[1:]
    
    plt.figure(figsize=(10,6))
    for y in ys[:-1]:
        plt.scatter(df[x1],df[y],c=df[ys[-1]],label=y)
    
    set_figstyle(0,df.shape[0],1300,3000,'sequential_number (k)','acceleration',True)
    plt.gca().set_xticklabels([int(x/1000) for x in plt.gca().get_xticks()])

def plot_data_p1(df,features):
    
    x1 = features[0]
    ys = features[1:]
    
    plt.figure(figsize=(10,6))
    for y in ys[:-1]:
        plt.scatter(df[x1],df[y],c=df[ys[-1]])
    
    set_figstyle(0,df.shape[0],1300,3000,'sequential_number (k)','acceleration',True)
    plt.gca().set_xticklabels([int(x/1000) for x in plt.gca().get_xticks()])
    plt.text(10*1000,2800,'class 1')
    plt.text(33*1000,2800,'2')
    plt.text(38*1000,2800,'3')
    plt.text(50*1000,2800,'4')
    plt.text(70*1000,2800,'5')
    plt.text(77*1000,2800,'6')
    plt.text(120*1000,2800,'7')

    plt.text(10*1000, 2500, 'y')
    plt.text(10*1000, 2250, 'z')
    plt.text(10*1000, 1750, 'x')

def plot_data4EachClass(df, features):
    
    """
        parameters:
        ===========
        df: DataFrame for each participant
        features: column name in the DataFrame
        
        Output:
        =======
        divide the data based on class and plot each figure for each class
        """
    
    x1 = features[0]
    ys = features[1:]
    
    fig, axs = plt.subplots(3, 3, figsize=(14,12))
    plt.tight_layout()
    
    
    for i in range(1,8):
        class_label = i
        
        df_class = df[df['label'].values == i]
        
        if i>7:
            break
        
        plt.subplot(3,3,i)
        
        plt.subplots_adjust(hspace = 0.3, wspace=0.3)
        
        for y in ys[:-1]:
            plt.scatter(df_class[x1].values,df_class[y].values,label=y)
        

        plt.gca().set_xticklabels([int(x/1000) for x in plt.gca().get_xticks()])
        plt.title('class'+  str(class_label))
        plt.xlabel('sequential_number (k)')
        plt.ylabel('acceleration')
        plt.legend()


def Fast_Fourier_Transformation(time_interval,st):
    
    # max frequency is determined by the smallest time interval
    
    # preprocess the data
    # prepare the signa normalized to within [-1,1]
    
    st_norm = st - st.mean()
    st = np.array(st_norm/st_norm.max())
    
    
    # FFT
    f_max  = 1/time_interval
    N_mesh = st.size
    freq = np.linspace(0, f_max-0.1, N_mesh)
    fft  = np.fft.fft(st)
    
    # plot
    plt.plot(freq[:N_mesh//2],abs(fft.real)[:N_mesh//2])
    plt.ylabel('Amplitude',fontsize=16)
    plt.xlabel('Frequency [Hz]',fontsize=16)
    
    fcs_sort = []
    for i in range(N_mesh//2):
        fcs_sort.append(freq[np.argsort(-abs(fft.real[:N_mesh//2]))[i]])
    
    return fcs_sort



def get_Energy_Moments_FFT(time_interval,st,k):
    
    """
        :st  -> signal as a function of time
        :k   -> highest order
        
        return
        ======
        the n-order energy weighted sum, normalized to the highest order moment:
        
        M_n = sum_i  E^n_i * fft(E_i)
        
        """
    
    # max frequency is determined by the smallest time interval
    #time_interval = t[1] - t[0]
    
    f_max = 1/time_interval
    N_mesh = st.size
    
    freq = np.linspace(0, f_max, N_mesh)
    fft = np.fft.fft(st)
    
    moments = []
    
    
    for k in range(1,k+1):
        moment_k = 0
        for i in range(N_mesh//2):
            moment_k += freq[i]**k*abs(fft.real[i])
        moments.append(moment_k)
     

    return moments  


def get_features(dfs, window_size, step_size, time_interval):
    
    # concat the data for all the 15 participants
    df = pd.concat(dfs[i] for i in range(0,len(dfs)))
    
    segments = []
    
    for class_label in range(1,8):
        
        df_class = df[df['label'].values == class_label]
        
        assert len(df_class) > window_size
        
        for i in range(0, len(df_class) - window_size, step_size):
            
            start = i
            end   = i + window_size
            
            x_seg = df_class['x_acceleration'].values[start: end]
            y_seg = df_class['y_acceleration'].values[start: end]
            z_seg = df_class['z_acceleration'].values[start: end]
            
            
            x_seg_norm = x_seg - x_seg.mean()
            x_seg_st   = np.array(x_seg_norm/x_seg_norm.max())
            
            y_seg_norm = y_seg - y_seg.mean()
            y_seg_st   = np.array(y_seg_norm/y_seg_norm.max())
            
            z_seg_norm = z_seg - z_seg.mean()
            z_seg_st   = np.array(z_seg_norm/z_seg_norm.max())
            
            x_moments = get_Energy_Moments_FFT(time_interval,x_seg_st,3) #.reshape(1,3)  # order is set to be 3
            y_moments = get_Energy_Moments_FFT(time_interval,y_seg_st,3) #.reshape(1,3)  # order is set to be 3
            z_moments = get_Energy_Moments_FFT(time_interval,z_seg_st,3) #.reshape(1,3)  # order is set to be 3
            
            
            segments.append([x_seg.mean(), y_seg.mean(), z_seg.mean(),
                             x_seg.var(), y_seg.var(), z_seg.var(),
                             x_seg.max(), y_seg.max(), z_seg.max(),
                             x_seg.min(), y_seg.min(), z_seg.min(),
                             pd.DataFrame(x_seg).mad(), pd.DataFrame(y_seg).mad(),
                             pd.DataFrame(z_seg).mad(), 
                             x_seg.mean()**2+y_seg.mean()**2+z_seg.mean()**2,
                             np.corrcoef(x_seg.squeeze(),y_seg.squeeze())[0,1],
                             np.corrcoef(y_seg.squeeze(),z_seg.squeeze())[0,1],
                             np.corrcoef(z_seg.squeeze(),x_seg.squeeze())[0,1],
                             skew(x_seg)[0], skew(y_seg)[0], skew(z_seg)[0],
                             kurtosis(x_seg)[0], kurtosis(y_seg)[0], kurtosis(z_seg)[0],
                             x_moments[0],x_moments[1],x_moments[2],
                             y_moments[0],y_moments[1],y_moments[2],
                             z_moments[0],z_moments[1],z_moments[2], 
                             class_label])

    return segments


def get_features_balanced(dfs, window_size_array, step_size_array, time_interval):
    
    """
        time_interval: 1/52
        """
    
    # concat the data for all the 15 participants
    df = pd.concat(dfs[i] for i in range(0,len(dfs)))
    
    segments = []
    
    for class_label in range(1,8):
        
        df_class = df[df['label'].values == class_label]
        
        window_size = window_size_array[class_label-1]
        step_size   = step_size_array[class_label-1]
        
        assert len(df_class) > window_size
        
        sample_size = 0
        for i in range(0, len(df_class) - window_size, step_size):
            
            sample_size +=1
            
            start = i
            end   = i + window_size
            
            x_seg = df_class['x_acceleration'].values[start: end]
            y_seg = df_class['y_acceleration'].values[start: end]
            z_seg = df_class['z_acceleration'].values[start: end]
            
            x_seg_norm = x_seg - x_seg.mean()
            x_seg_st   = np.array(x_seg_norm/x_seg_norm.max())
            
            y_seg_norm = y_seg - y_seg.mean()
            y_seg_st   = np.array(y_seg_norm/y_seg_norm.max())
            
            z_seg_norm = z_seg - z_seg.mean()
            z_seg_st   = np.array(z_seg_norm/z_seg_norm.max())
            
            x_moments = get_Energy_Moments_FFT(time_interval,x_seg_st,3) #.reshape(1,3)  # order is set to be 3
            y_moments = get_Energy_Moments_FFT(time_interval,y_seg_st,3) #.reshape(1,3)  # order is set to be 3
            z_moments = get_Energy_Moments_FFT(time_interval,z_seg_st,3) #.reshape(1,3)  # order is set to be 3
            
            
            segments.append([x_seg.mean(), y_seg.mean(), z_seg.mean(),
                             x_seg.var(), y_seg.var(), z_seg.var(),
                             x_seg.max(), y_seg.max(), z_seg.max(),
                             x_seg.min(), y_seg.min(), z_seg.min(),
                             pd.DataFrame(x_seg).mad(), pd.DataFrame(y_seg).mad(),
                             pd.DataFrame(z_seg).mad(),
                             # x_seg.sum(),y_seg.sum(),z_seg.sum(), # removed
                             x_seg.mean()**2+y_seg.mean()**2+z_seg.mean()**2,
                             np.corrcoef(x_seg.squeeze(),y_seg.squeeze())[0,1],
                             np.corrcoef(y_seg.squeeze(),z_seg.squeeze())[0,1],
                             np.corrcoef(z_seg.squeeze(),x_seg.squeeze())[0,1],
                             skew(x_seg)[0], skew(y_seg)[0], skew(z_seg)[0],
                             kurtosis(x_seg)[0], kurtosis(y_seg)[0], kurtosis(z_seg)[0],
                             x_moments[0],x_moments[1],x_moments[2],
                             y_moments[0],y_moments[1],y_moments[2],
                             z_moments[0],z_moments[1],z_moments[2],
                             class_label])
        
        print("class: {}, window_size: {}, step_size: {}, sample_size: {}".format(class_label,window_size,step_size,sample_size))

    return segments


# for 15 participants

def get_segments4All(dfs, window_size, step_size, progress_bar):
    
    segments = []
    labels = []
    
    
    df = pd.concat(dfs[p] for p in range(0,len(dfs)))
    
    # loop over class [1,2,..,7]
    
    
    for class_label in range(1,8):
        
        print("Processing the data of class: {}\n".format(class_label))
        
        df_class = df[df['label'].values == class_label]
        assert len(df_class) > window_size
        
        
        # for each  given class, slide the window
        #for i in range(0,len(df_class)- window_size, step_size):
        
        # progress bar
        if  progress_bar:
            f = IntProgress(min=0, max=len(df_class)) # instantiate the bar
            display(f) # display the bar
        
        for i in range(0,len(df_class) - window_size, step_size):
            start = i
            end   = i + window_size
            xseg = df_class['x_acceleration'].values[start: end]
            yseg = df_class['y_acceleration'].values[start: end]
            zseg = df_class['z_acceleration'].values[start: end]
            segments.append([xseg, yseg, zseg])
            labels.append(class_label)
            
            if  progress_bar:
                f.value += step_size # signal to increment the progress bar



    return labels,segments


def preprocess_data(dfs, window_size, step_size, num_feature):
    
    labels4all,segments4all = get_segments4All(dfs, window_size, step_size, False)
    segments4all_array      = np.array(segments4all, dtype=np.float64).reshape(-1,num_feature, window_size)
    
    return labels4all, segments4all_array


def decision_trees_gridsearch(X_train,y_train,X_test,y_test,criterions,max_depths, min_samples_leaves):
    
    F1_weighted = 0.0
    best_accuracy = 0.0
    
    results = []
    
    for  criterion in criterions:
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leaves:
                
                tree_clf = make_pipeline(RobustScaler(),DecisionTreeClassifier(   \
                                                                criterion=criterion,
                                                                max_depth=max_depth,
                                                                min_samples_leaf=min_samples_leaf)).fit(X=X_train,y=y_train)
                y_pred = tree_clf.predict(X_test)
                F1_temp = f1_score(y_test,y_pred,average='weighted')
                accuracy_tmp = metrics.accuracy_score(y_test, y_pred)
                results.append([criterion,max_depth,min_samples_leaf,F1_temp,accuracy_tmp])
                                                                
                # save the best case based on F1 weighted score
                #if F1_temp*accuracy_tmp/(F1_temp+accuracy_tmp) > F1_weighted*best_accuracy/(F1_weighted*best_accuracy+0.00001):
                if F1_temp > F1_weighted:
                    F1_weighted = F1_temp
                    tree_clf_best  = tree_clf
                    best_parameter = [criterion,max_depth,min_samples_leaf]
                    cm_best = metrics.confusion_matrix(y_test,y_pred)
                    print('criterion: {}, max_depth: {}, min_samples_leaf: {}, F1_temp: {}, accuracy_tmp: {}'.format(criterion,max_depth,min_samples_leaf,F1_temp,accuracy_tmp))


    return results,tree_clf_best,cm_best,best_parameter


def knn_gridsearch(X_train,y_train,X_test,y_test,n_neighbors_range):
    
    
    
    # empty list that will hold cv scores
    F1_weighted   = 0.0
    best_accuracy = 0.0
    scores        = []
    
    for  k  in n_neighbors_range:
        
        knn_clf = make_pipeline(RobustScaler(),KNeighborsClassifier(n_neighbors=k)).fit(X=X_train, y=y_train)
        y_pred       = knn_clf.predict(X_test)
        F1_temp      = f1_score(y_test,y_pred,average='weighted')
        accuracy_tmp = metrics.accuracy_score(y_test, y_pred)
        scores.append([k,F1_temp,accuracy_tmp])
                                
        if F1_temp > F1_weighted:
            F1_weighted    = F1_temp
            knn_clf_best   = knn_clf
            best_parameter = k
            cm_best        = metrics.confusion_matrix(y_test,y_pred)

            print('k: {}, F1_score: {}, accuracy: {}'.format(k,F1_weighted,accuracy_tmp))


    return scores,knn_clf_best,cm_best,best_parameter


def handcrafted_model(model, dfs, ratios, window_size, features, time_interval, overlap):
    
    """
        model: RandomForestClassifier()
        """
    
    # set hyperparameters for the window
    window_size_array    =  np.multiply(ratios, window_size)   # times of 0.5 second
    if overlap:
        step_size_array      =  np.floor(window_size_array//2+1) # half overlap
        filename             = 'HAR_Processed_Data_X34Y1_ovlp'+str(window_size)+'.csv'
    else:
        step_size_array      =  window_size_array  # same as window_size and thus no overlaping between samples
        filename             = 'HAR_Processed_Data_X34Y1_no_ovlp'+str(window_size)+'.csv'
        
    step_size_array = step_size_array.astype(int)
    
    # generate balanced samples
    segments_array_balanced = np.array(get_features_balanced(dfs, window_size_array, step_size_array, time_interval))

    segments_balanced_df = pd.DataFrame(segments_array_balanced)
    print(segments_balanced_df.shape, len(features))

    segments_balanced_df.columns = features
    segments_balanced_df['label'] = segments_balanced_df['label'].map(int)
    
    
    
    segments_balanced_df.to_csv(filename)
    
    X = segments_balanced_df.iloc[:,:-1].values
    y = segments_balanced_df['label'].iloc[:].values
                                   
    # scale the X
    scaler_pipeline = Pipeline([('robust_scaler',RobustScaler())])

    segments_scaled = []
    segments_scaled.append(scaler_pipeline.fit_transform(X))
    X_scaled = np.array(segments_scaled).squeeze()
                                   
                                   
    # random state is choosen as 10% of each sample size
    random_state = int(segments_balanced_df.shape[0]/70)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2,
                                                        random_state = random_state )
                                   
    model_clf = make_pipeline(RobustScaler(),model).fit(X=X_train, y=y_train)
    y_pred   = model_clf.predict(X_test)
    model_clf_cm = metrics.confusion_matrix(y_test,y_pred)
                                   
    return window_size, model_clf_cm


def model_simple(num_feature,window_size):
    
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, (2,2), activation='relu', input_shape=(num_feature,window_size,1))) # 64 nodes, 3x3 filter matrix
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(36, (2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    return model



def model_CX(num_feature,window_size):
    """
        
        """
    
    model = Sequential()
    model.add(Conv2D(18, (2, 12), padding='same', input_shape=(num_feature,window_size,1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2D(36, (1, 12), strides=2, padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2D(24, (1, 12), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    
    return model


# precision is the fraction of events where we correctly declared 𝑖
# out of all instances where the algorithm declared 𝑖.
# Conversely, recall is the fraction of events where we correctly declared 𝑖
# out of all of the cases where the true of state of the world is 𝑖



def precision(label, confusion_matrix):
    
    # precision for each class
    col = confusion_matrix[:, label]
    
    if float(col.sum()) < 0.001:
        return 0
    
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    
    # recall for each class
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def accuracy_macro_average(confusion_matrix):
    correct = 0
    total  = 0
    
    for i in range(0, confusion_matrix.shape[0]):
        correct += confusion_matrix[i,i]
        
        for j in range(0, confusion_matrix.shape[0]):
            total += confusion_matrix[j,i]

    return correct / total


def precision_macro_average(confusion_matrix):
    
    # mean precision for all classes
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    
    # mean recall for all classes
    rows, columns = confusion_matrix.shape
    
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def f1_score_macro_average(confusion_matrix):
    
    precision = precision_macro_average(confusion_matrix)
    recall    = recall_macro_average(confusion_matrix)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




def plot_contour(x,y,z,xrange,yrange,zrange,xlabel,ylabel,zlabel,cmap):
    
    fs = 18
    fig = plt.figure(figsize=(8,6))
    #minorLocator = MultipleLocator(1)
    
    # grid/mesh
    xi2 = np.linspace(x.min(), x.max(), 1000)
    yi2 = np.linspace(y.min(), y.max(), 1000)
    zi2 = griddata(x, y, z, xi2, yi2, interp='linear')
    
    
    levels2 = [0.0,0.2,0.4,0.6,0.8,1.0] #np.arange(zrange[0],zrange[1],zrange[2])
    
    ax0 = plt.gca()
    NormNul = ax0.contourf(xi2, yi2, zi2, levels2, cmap=cmap)
    ax0.set(xlim=xrange,ylim=yrange)
    
    ax0.set_xlabel(xlabel,size=20)
    ax0.set_ylabel(ylabel,size=20)
    cbar0 = fig.colorbar(NormNul,shrink=0.8,aspect=10)
    cbar0.set_label(zlabel, size=fs)
    cbar0.ax.tick_params(labelsize=fs)
    
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.tick_params(which='major',direction='in',length=10,labelsize=20)
    plt.tick_params(which='minor',direction='in',length=5,labelsize=20)


def f1(y_true, y_pred):
    """
        computing macro f1 score in Keras models
    """
    
    def recall(y_true, y_pred):
        
        """Recall metric.
            Only computes a batch-wise average of recall.
            
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.
            
            Only computes a batch-wise average of precision.
            
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall    = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def CompressedData_PCA(X, y):
    
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    
    df4pca = pd.DataFrame(X,columns=feat_cols)
    df4pca['label'] = y
    df4pca['label'] = df4pca['label'].apply(lambda i: str(i))
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)
    
    df4pca['pca-one'] = pca_result[:,0]
    df4pca['pca-two'] = pca_result[:,1]
    df4pca['pca-three'] = pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    return df4pca

def plot_ggplot4classification(df,xaxis,yaxis,zaxis):
    
    chart = ggplot(df, aes(x=xaxis, y=yaxis, color=zaxis) ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("Principal Components colored by digit")
    
    return chart


def plot_ggplot4classification_xyrange(df,xaxis,yaxis,zaxis, x_min, x_max, y_min, y_max):
    
    chart = ggplot(df, aes(x=xaxis, y=yaxis, color=zaxis) ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("Principal Components colored by digit") \
        + xlim(x_min, x_max)   \
        + ylim(y_min, y_max)
    
    return chart

def print_scores(cm,y_test,y_pred):
    print("accuracy: {:.3f}".format(accuracy_macro_average(cm)))
    print("precision:  {:.3f}".format(precision_macro_average(cm)))
    print("recall:  {:.3f}".format(recall_macro_average(cm)))
    print("F1 score (macro):  {:.3f}".format(f1_score_macro_average(cm)))
    print("F1 score (weighted): {:.3f}".format(f1_score(y_test,y_pred,average='weighted')))


def confidence_interval4proportion(p, sample_size, alpha):
    assert sample_size >0
    
    mean     = p
    variance = p*(1-p)
    
    statistic = mean/(variance/sqrt(sample_size))
    
    
    interval = norm.ppf(statistic)
    
    return [mean - interval, mean + inverval]


def statistic_compare_two_models(p1,p2,n1,n2,alpha,sides):
    
    
    """
        propotional test for the same sample size.
        
        p1 -- average score for model1
        p2 -- average score for model2
        alpha -- confidence level
        sides -- 1 for one-tail; 2 for two-tail test
        """
    
    # avoid singularity problem
    
    assert n1 > 0.1  and n2 > 0.1
    
    # mean proportion
    prob = (p1*n1 + p2*n2)/ (n1 + n2) #  or mean = B_conv_rate
    
    # standard error
    SE = np.sqrt(prob*(1-prob)*(1/n1 + 1/n2))
    
    stats = abs(p1-p2)/SE
    
    p_value = norm.sf(abs(stats))*sides # sf(x, loc=0, scale=1) returns the p value
    
    if p_value > alpha:
        print('Fail to reject H0: Cannot distinguish the performance of the two models at the confidence level of {}%'.format(100*(1-alpha)))
    else:
        print('Reject H0: The performance of the two models is different at the confidence level of {}%'.format(100*(1-alpha)))

    return stats,p_value


def recall_eachClass(confusion_matrix):
    
    propotion_CI = []
    
    for i in range(0, confusion_matrix.shape[0]):
        class_total = 0
        for j in range(0, confusion_matrix.shape[0]):
            class_total += confusion_matrix[i,j]
        CI = sm.stats.proportion_confint(confusion_matrix[i,i], class_total, alpha = 0.05)
        propotion = confusion_matrix[i,i]/class_total
        propotion_CI.append([i+1,propotion,CI[0],CI[1]])
    
    columnName = ['Label_r','recall','recall_Low','recall_Up']
    df_P_CI = pd.DataFrame.from_records(propotion_CI, columns = columnName)
    
    return df_P_CI


def precision_eachClass(confusion_matrix):
    
    propotion_CI = []
    
    for i in range(0, confusion_matrix.shape[0]):
        class_total = 0
        for j in range(0, confusion_matrix.shape[0]):
            class_total += confusion_matrix[j,i]
        
        CI = sm.stats.proportion_confint(confusion_matrix[i,i], class_total, alpha = 0.05)
        propotion = confusion_matrix[i,i]/class_total
        propotion_CI.append([i+1,propotion,CI[0],CI[1]])
    
    columnName = ['Label_p','precision','precision_Low','precision_Up']
    df_P_CI = pd.DataFrame.from_records(propotion_CI, columns = columnName)
    
    return df_P_CI

def print_score4each(cm):
    precision4each = precision_eachClass(cm)
    recall4each    = recall_eachClass(cm)
    score4each = pd.concat([precision4each,recall4each], axis=1)
    score4each.drop(['Label_r'], axis=1, inplace=True)
    return score4each


def check_overfitting_rf(X_train,y_train,X_test,y_test,n_estimators, \
                          max_depth, min_samples_split, metric_max):
    
    """
    Input:
    - X_train: features of training data
    - y_train: class label of training data
    - X_test:  features of test data
    - y_test:  class label of test data
    
    Model parameters
    
    
    - n_estimators: number of trees in the foreset 
    - max_depth: max number of levels in each decision tree
    - min_samples_split: min number of data points placed in a node before the node is split
    
    Output:
    - metric_max: maximal metric value
    """
    
    rf_clf = make_pipeline(RobustScaler(),
                           RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split)).fit(X=X_train, y=y_train)
                               
    y_pred4test    = rf_clf.predict(X_test)
    y_pred4train   = rf_clf.predict(X_train)
    

    f14train = f1_score(y_train, y_pred4train, average='weighted')
    f14test  = f1_score(y_test, y_pred4test, average='weighted')

    metric = f14test - 1.5*abs(f14train-f14test)  #  play with the weights

    if metric > metric_max:
        print("n_estimators: {}, max_depth:{}, min_samples_split:{}" "\n"
              "f1(weighted)/Train:{:.2f}, f1(weighted)/Test:{:.2f}, metric:{:.2f}".format(n_estimators,
                                                                                          max_depth,
                                                                                          min_samples_split,
                                                                                          f14train,f14test,metric))
        metric_max = metric
    return metric_max

def xgboost_fit_predict(X_train,y_train,X_test,y_test):
    
    xgb_reg = xgboost.XGBClassifier(n_estimators=1000, random_state=42)
    
    xgb_reg = xgb_reg.fit(X_train, y_train,
                          eval_set=[[X_train, y_train],[X_test, y_test]],
                          verbose=100,
                          early_stopping_rounds=10)

    y_pred = xgb_reg.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    print("Validation MSE:", val_error)

    return xgb_reg

def check_overfitting_gb(X_train,y_train,X_test,y_test,n_estimators, \
                          max_depth, min_samples_split, metric_max):
    
    """
    Input:
    - X_train: features of training data
    - y_train: class label of training data
    - X_test:  features of test data
    - y_test:  class label of test data
    
    Model parameters
    
    
    - n_estimators: number of trees in the foreset 
    - max_depth: max number of levels in each decision tree
    - min_samples_split: min number of data points placed in a node before the node is split
    
    Output:
    - metric_max: maximal metric value
    """
    
    rf_clf = make_pipeline(RobustScaler(),
                           GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split)).fit(X=X_train, y=y_train)
                               
    y_pred4test    = rf_clf.predict(X_test)
    y_pred4train   = rf_clf.predict(X_train)
    

    f14train = f1_score(y_train, y_pred4train, average='weighted')
    f14test  = f1_score(y_test, y_pred4test, average='weighted')

    metric = f14test - 1.5*abs(f14train-f14test)  #  play with the weights

    if metric > metric_max:
        print("n_estimators: {}, max_depth:{}, min_samples_split:{}" "\n"
              "f1(weighted)/Train:{:.2f}, f1(weighted)/Test:{:.2f}, metric:{:.2f}".format(n_estimators,
                                                                                          max_depth,
                                                                                          min_samples_split,
                                                                                          f14train,f14test,metric))
        metric_max = metric
    return metric_max


