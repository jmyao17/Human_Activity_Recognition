
import pandas as pd
import numpy as np

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from matplotlib.mlab import griddata


from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier


import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.models import Sequential
from keras import backend as K
import keras_metrics as km

from ipywidgets import IntProgress
from IPython.display import display
import time
import ggplot
from ggplot import *

import statsmodels.api as sm

from scipy.stats import norm


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


def get_features(dfs, window_size, step_size):
    
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
            
            segments.append([x_seg.mean(), y_seg.mean(), z_seg.mean(),
                             x_seg.var(), y_seg.var(), z_seg.var(),
                             x_seg.max(), y_seg.max(), z_seg.max(),
                             x_seg.min(), y_seg.min(), z_seg.min(),
                             pd.DataFrame(x_seg).mad(), pd.DataFrame(y_seg).mad(),
                             pd.DataFrame(z_seg).mad(),
                             class_label])

    return segments


def get_features_balanced(dfs, window_size_array, step_size_array):
    
    
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
            
            segments.append([x_seg.mean(), y_seg.mean(), z_seg.mean(),
                             x_seg.var(), y_seg.var(), z_seg.var(),
                             x_seg.max(), y_seg.max(), z_seg.max(),
                             x_seg.min(), y_seg.min(), z_seg.min(),
                             pd.DataFrame(x_seg).mad(), pd.DataFrame(y_seg).mad(),
                             pd.DataFrame(z_seg).mad(),
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
    minorLocator = MultipleLocator(1)
    
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
