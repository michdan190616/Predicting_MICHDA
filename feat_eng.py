import numpy as np

def cut_range(column,lower_bnd, upper_bnd):
    """
    A function that takes as inputs:
    - coloumn: a coloumn of the dataset
    - lower_bnd: a lower bound
    - upper_bnd: an upper bound
    
    And returns the same coloumn, where every value smaller than lower_bnd and bigger than
    upper bound has been set to Nan

    """

    column = np.array(column)

    idx_lower = column<lower_bnd
    idx_upper = column>upper_bnd

    column[idx_lower] = np.nan
    column[idx_upper] = np.nan

    return column

#############------------------------------#############

def replace_nan_with_mode(X):
    """
    A function that takes as input:
    - X: the subset of categorical features of the dataset
    
    And runs through the coloumns of X setting each Nan value to the mode of the feature.

    The function returns the modified X.

    """

    nan_indices = np.isnan(X)

    nan_mask = np.isnan(X)
    no_nan_mask = ~nan_mask

    modes = np.empty(X.shape[1])
    for col in range(X.shape[1]):
        non_nan_values = X[no_nan_mask[:, col], col]
        if non_nan_values.size == 0:
            modes[col] = np.nan
        else:
            modes[col] = np.argmax(np.bincount(non_nan_values.astype(int)))

    for row, col in np.transpose(np.where(nan_indices)):
        X[row, col] = modes[col]

    return X

#############------------------------------#############

def replace_nan_with_median(X):
    """
    A function that takes as input:
    - X: the subset of numeric features of the dataset
    
    And runs through the coloumns of X setting each Nan value to the median of the feature.

    The function returns the modified X.

    """


    column_medians = np.nanmedian(X, axis=0)
    
    nan_indices = np.isnan(X)

    for col in range(X.shape[1]):
        X[nan_indices[:, col], col] = column_medians[col]

    return X

#############------------------------------#############

def winsorize_matrix(matrix, lower_percentile=0.05, upper_percentile=0.95):
    """
    A function that takes as inputs:
    - matrix: the subset of numeric features of the dataset
    - lower_percentile: the lower quantile for the winsorization, set at 5%
    - upper_percentile: the bigger quantile for the winsorization, set at 95%
    
    And winsorizes the columns of matrix by setting to the 5^th quantile all the values lower than 5^th qunatile,
    and to the 95^th quantile all the values bigger than the 95^th quantile.

    The function returns the winsorized matrix.

    """


    lower_limit = np.percentile(matrix, lower_percentile * 100, axis=0)
    upper_limit = np.percentile(matrix, upper_percentile * 100, axis=0)

    winsorized_matrix = np.copy(matrix)
    for col in range(matrix.shape[1]):
        winsorized_matrix[:, col] = np.clip(matrix[:, col], lower_limit[col], upper_limit[col])

    return winsorized_matrix

#############------------------------------#############

def standardize(x):
    """
    A function that takes as input:
    - x: the subset of numeric features of the dataset
    
    And standardize the features by subtracting to each column its mean, and dividing by its standard deviation.

    The function returns the standardized x, called std_data.

    """

    mean=np.mean(x, axis=0)
    std=np.std(x, axis=0)
    std_data=(x-mean)/std
    return std_data

#############------------------------------#############

def create_cat(x,name,names):
    """
    A function that takes as input:
    - X: the subset of categorical features of the dataset
    - name: the name of a feature
    - names: the np.array of feature names
    
    And creates the dummy variable for each class in the name column. The function also deletes name from the
    names array, to then add to the same array one name for each newly created dummy variable.

    The function returns the modified X and the array of feature names.

    """

    col=x[:,names==name]

    x=np.delete(x,np.argmax(names==name),axis=1)
    names=np.delete(names,np.argmax(names==name))
    values=np.unique(col)

    for val in values[:-1]:
        x=np.column_stack((x,np.array(col==val,dtype=int)))
        names=np.append(names,name+f'_{val}')

    return x,names

#############------------------------------#############

def create_polynomial_features(X, degree, sq_root):
    """
    A function that takes as input:
    - X: the subset of numeric features of the dataset
    - degree: the highest desired degree of the polynomial features
    - sq_root: a bynary value that takes value True to generate also the degree=1/2 polynomial feature, 0 otherwise
    
    And after exponentiating the columns of X to the desired degree(s), it adds them back to X.

    The function returns the modified X, called polynomial_features.

    """

    polynomial_features = []

    for d in range(1, degree + 1):

        polynomial_features.append(X**d)

    if(sq_root):

        polynomial_features.append(np.sqrt(np.abs(X)))

    return np.hstack(polynomial_features)

#############------------------------------#############

def feat_engineering(X, col_names, d, sq_root):
    """
    A function that takes as input:
    - X: the subset of numeric features of the dataset
    - col_names: the array of feature names
    - d: the highest desired degree of the polynomial features
    - sq_root: a bynary value that takes value True to generate also the degree=1/2 polynomial feature, 0 otherwise
    
    And carries out the feature engineering of the dataset. First, it deletes all features with more than 25% of Nan,
    then it deletes all the columns not pertinent to thre analysis. Subsequently, it calls previously defined functions
    to correclty manage both numeric and categorical features, also adding a constant to the dataset.

    The function returns the processed dataset, called df_final_poly.

    """


    col_names = np.array(col_names)

    #drop columns with more than 25% of nans

    idx_25=np.isnan(X).sum(axis=0)/len(X)<0.25   
    data_25=X[:,idx_25]
    names_25=col_names[idx_25]

    #select column not pertinent to our purpose

    delete_ = ['_STATE','FMONTH','IDATE','IMONTH','IDAY','IYEAR','DISPCODE','SEQNO','_PSU','PERSDOC2',
           'MARITAL','EDUCA','RENTHOM1','VETERAN3','CHILDREN','INCOME2','INTERNET','QLACTLM2','BLIND',
           'DIFFWALK','DIFFDRES','DIFFALON','STOPSMK2','LASTSMK2','ALCDAY5','FRUITJU1','FVBEANS','SEATBELT',
           'QSTVER','QSTLANG','_STSTR','_STRWT', '_RAWRAKE', '_WT2RAKE','_DUALUSE','_LLCPWT','_LTASTH1','_CASTHM1', 
           '_ASTHMS1','_DRDXAR1','_MRACE1', '_HISPANC', '_RACE', '_RACEG21','_RACEGR3', '_RACE_G1', '_AGEG5YR',
           '_AGE_G','HTIN4', 'HTM4', 'WTKG3','_BMI5CAT', 'DRNKANY5', 
           'DROCDY3_', 'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_',
           'VEGEDA1_', '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP', '_FRT16', '_VEG23',
           '_FRUITEX', '_VEGETEX','PAMISS1_','_PA30021', '_LMTACT1',
           '_LMTWRK1', '_LMTSCL1', '_RFSEAT2', '_RFSEAT3']

    idx_clean=[]
    for i in names_25:
        idx_clean.append(i not in delete_)

    names_clean=names_25[idx_clean]
    data_clean=data_25[:,idx_clean]

    data_prova_cut_range = data_clean.copy()

    #fixing of columns with unreasonable values

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='WEIGHT2']<=999) & (data_prova_cut_range[:,names_clean=='WEIGHT2']>=50)).ravel(),names_clean=='WEIGHT2']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='WEIGHT2']<=999) & (data_prova_cut_range[:,names_clean=='WEIGHT2']>=50)).ravel(),names_clean=='WEIGHT2']*0.453592+9000
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='HEIGHT3']<=711) & (data_prova_cut_range[:,names_clean=='HEIGHT3']>=200)).ravel(),names_clean=='HEIGHT3']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='HEIGHT3']<=999) & (data_prova_cut_range[:,names_clean=='HEIGHT3']>=50)).ravel(),names_clean=='HEIGHT3']//100*30.48+data_prova_cut_range[((data_prova_cut_range[:,names_clean=='HEIGHT3']<=999) & (data_prova_cut_range[:,names_clean=='HEIGHT3']>=50)).ravel(),names_clean=='HEIGHT3']%100*2.54+9000

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='MENTHLTH']==88).ravel(),names_clean=='MENTHLTH']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='CHECKUP1']==8).ravel(),names_clean=='CHECKUP1']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='FRUIT1']==555).ravel(),names_clean=='FRUIT1']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='FVGREEN']==555).ravel(),names_clean=='FVGREEN']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='FVORANG']==555).ravel(),names_clean=='FVORANG']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='VEGETAB1']==555).ravel(),names_clean=='VEGETAB1']=0

    data_prova_cut_range[(data_prova_cut_range[:,names_clean=='STRENGTH']==888).ravel(),names_clean=='STRENGTH']=0

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=199) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=101)).ravel(),names_clean=='FRUIT1']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=199) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=101)).ravel(),names_clean=='FRUIT1']-100)*30
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=299) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=201)).ravel(),names_clean=='FRUIT1']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=299) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=201)).ravel(),names_clean=='FRUIT1']-200)*4
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=399) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=300)).ravel(),names_clean=='FRUIT1']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FRUIT1']<=399) & (data_prova_cut_range[:,names_clean=='FRUIT1']>=300)).ravel(),names_clean=='FRUIT1']-300)

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=199) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=101)).ravel(),names_clean=='FVGREEN']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=199) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=101)).ravel(),names_clean=='FVGREEN']-100)*30
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=299) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=201)).ravel(),names_clean=='FVGREEN']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=299) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=201)).ravel(),names_clean=='FVGREEN']-200)*4
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=399) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=300)).ravel(),names_clean=='FVGREEN']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVGREEN']<=399) & (data_prova_cut_range[:,names_clean=='FVGREEN']>=300)).ravel(),names_clean=='FVGREEN']-300

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=199) & (data_prova_cut_range[:,names_clean=='FVORANG']>=101)).ravel(),names_clean=='FVORANG']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=199) & (data_prova_cut_range[:,names_clean=='FVORANG']>=101)).ravel(),names_clean=='FVORANG']-100)*30
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=299) & (data_prova_cut_range[:,names_clean=='FVORANG']>=201)).ravel(),names_clean=='FVORANG']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=299) & (data_prova_cut_range[:,names_clean=='FVORANG']>=201)).ravel(),names_clean=='FVORANG']-200)*4
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=399) & (data_prova_cut_range[:,names_clean=='FVORANG']>=300)).ravel(),names_clean=='FVORANG']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='FVORANG']<=399) & (data_prova_cut_range[:,names_clean=='FVORANG']>=300)).ravel(),names_clean=='FVORANG']-300

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=199) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=101)).ravel(),names_clean=='VEGETAB1']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=199) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=101)).ravel(),names_clean=='VEGETAB1']-100)*30
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=299) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=201)).ravel(),names_clean=='VEGETAB1']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=299) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=201)).ravel(),names_clean=='VEGETAB1']-200)*4
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=399) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=300)).ravel(),names_clean=='VEGETAB1']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='VEGETAB1']<=399) & (data_prova_cut_range[:,names_clean=='VEGETAB1']>=300)).ravel(),names_clean=='VEGETAB1']-300

    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='STRENGTH']<=199) & (data_prova_cut_range[:,names_clean=='STRENGTH']>=101)).ravel(),names_clean=='STRENGTH']=(data_prova_cut_range[((data_prova_cut_range[:,names_clean=='STRENGTH']<=199) & (data_prova_cut_range[:,names_clean=='STRENGTH']>=101)).ravel(),names_clean=='STRENGTH']-100)*4
    data_prova_cut_range[((data_prova_cut_range[:,names_clean=='STRENGTH']<=299) & (data_prova_cut_range[:,names_clean=='STRENGTH']>=201)).ravel(),names_clean=='STRENGTH']=data_prova_cut_range[((data_prova_cut_range[:,names_clean=='STRENGTH']<=299) & (data_prova_cut_range[:,names_clean=='STRENGTH']>=201)).ravel(),names_clean=='STRENGTH']-200

    data_prova_cut_range[:,names_clean=='_VEGESUM']=data_prova_cut_range[:,names_clean=='_VEGESUM']/100
    data_prova_cut_range[:,names_clean=='_FRUTSUM']=data_prova_cut_range[:,names_clean=='_FRUTSUM']/100


    #setting to nan all the values that are not supposed to be in the dataset

    lower_bnd = [1, 1, 0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9000,9000,1,1,1,1,  0,  0,  0,  0,1,  0,1,1,1,1,1,1,1,1,1,1,18,   1,1,1,1,1,1,1,1,    1,1,    0,     0,1,1,1,  0,   0,    0,1,1,1,1,1,1,1,1]
    upper_bnd = [5,30,30,2,2,4,4,2,4,2,2,2,2,2,2,2,2,2,4,2,8,9998,9998,2,2,2,3,300,300,300,300,2,299,2,2,2,2,2,2,3,2,6,2,99,9999,2,6,4,5,4,2,2,98999,2,99998, 99998,2,2,2,501,8590,98999,4,2,3,3,2,4,2,2]

    for j in range(len(names_clean)):

        data_prova_cut_range[:,j] = cut_range(data_prova_cut_range[:,j],lower_bnd[j], upper_bnd[j])


    #see which variables are categorical and which are not

    names_cat=np.array(['GENHLTH','HLTHPLN1', 'MEDCOST',
       'CHECKUP1', 'BPHIGH4', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2',
       'CVDSTRK3', 'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1',
       'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'SEX',
        'EMPLOY1','USEEQUIP', 'DECIDE', 'SMOKE100', 'USENOW3',
        'EXERANY2','FLUSHOT6', 'PNEUVAC3', 'HIVTST6', '_RFHLTH',
        '_HCVU651','_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_PRACE1',
        '_AGE65YR','_RFBMI5', '_CHLDCNT','_EDUCAG','_INCOMG','_SMOKER3','_RFSMOK3', '_RFBING5','_RFDRHV5','_FRTLT1', '_VEGLT1','_TOTINDA',
        '_PACAT1', '_PAINDX1', '_PA150R2','_PA300R2','_PASTRNG', '_PAREC1','_PASTAE1','_AIDTST3'])
    
    names_num=names_clean[~np.isin(names_clean,names_cat)]

    data_prova_cut_range_cat=data_prova_cut_range[:,np.isin(names_clean,names_cat)]
    data_prova_cut_range_num=data_prova_cut_range[:,~np.isin(names_clean,names_cat)]
    
    #modify non categorical columns
    
    df_final_num=replace_nan_with_median(data_prova_cut_range_num)
    df_final_num=winsorize_matrix(df_final_num)
    df_final_num=standardize(df_final_num)

    #modify categorical columns

    df_final_cat=replace_nan_with_mode(data_prova_cut_range_cat)

    for name in names_cat:
        df_final_cat,names_cat=create_cat(df_final_cat,name,names_cat)

    #join categorical column and levels column and add constant term

    df_final_num_poly=create_polynomial_features(df_final_num,d, sq_root)
    df_final_poly=np.column_stack((df_final_num_poly,df_final_cat))

    df_final_poly=np.column_stack((np.ones(df_final_poly.shape[0]),df_final_poly))

    return df_final_poly

#############------------------------------#############

def split_dataset(x,y,k):
    """
    A function that takes as inputs:
    - x: the dataset
    - y: the array of dependent variables
    - k: the desired number of subsets

    And after randomly shuffling both x and y, it splits them in k different sets of training and test sets.

    The function returns 4 lists of k subsets, 2 for the training dataset and dependent variables (x_train, y_train)
    and 2 for the raining dataset and dependent variables (x_test, y_test)

    """
    y = np.array(y)
    #shuffle rows randomly
    idx = np.random.permutation(len(y))
    x=x[idx]
    y=y[idx]
    
    #split dataset in k folds
    x_split=np.array_split(x,k)
    y_split=np.array_split(y,k)
    
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    
    for i in range(k):
        x_test.append(x_split[i])
        y_test.append(y_split[i])
        x_train.append(np.concatenate(x_split[:i]+x_split[i+1:]))
        y_train.append(np.concatenate(y_split[:i]+y_split[i+1:]))
        
    return x_train,y_train,x_test,y_test