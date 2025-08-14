
""" Author: Benyamin Meschede-Krasa 
utility functions for inner speech for sequence production pipeline """
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
######################
######  PARAMS  ######
######################
ALL_CH = np.arange(256)

########################################################
######             UTILITY FUNCTIONS              ######
########################################################

def add_columns_groups(df,channel_sets, seq_columns):
    """Create multi-level column index to organize DataFrame by brain regions.
    
    This function restructures a DataFrame with neural channel data by creating
    a hierarchical column index that groups channels by brain region and organizes
    metadata columns into logical categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with neural data columns (starting with 'ch') and metadata.
    channel_sets : dict
        Mapping of brain region names to channel indices. Keys are region names,
        values are arrays of channel indices.
    seq_columns : list of str
        List of column names representing sequence position information (e.g.,
        'First', 'Second', 'Third') that should be grouped under 'SequencePosition'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchical column index where the first level groups
        columns by category (brain regions, 'SequencePosition', 'Misc') and
        second level contains the original column names. Columns are reordered
        to match the order: brain regions, SequencePosition, Misc.
        
    Notes
    -----
    - Channel columns (starting with 'ch') are mapped to brain regions based on
      their channel index
    - Channels not found in channel_sets are labeled as 'unknown'
    - The resulting DataFrame has a MultiIndex columns structure
    """

    # Mapping channel indices to area names
    channel_mapping = {}
    for area, indices in channel_sets.items():
        for idx in indices:
            channel_mapping[idx] = area
    
    column_groups =[]
    ch_cols = []
    misc_cols = []
    for col in df.columns:
        if col.startswith('ch'):
            ch_idx = int(col.split('_')[0][2:])
            column_groups.append(channel_mapping.get(ch_idx, 'unknown'))
            ch_cols.append(col)
        elif col in seq_columns:
            column_groups.append('SequencePosition')
        else:
            column_groups.append('Misc')
            misc_cols.append(col)

    df.columns = [column_groups,# top level of multiindex
                  df.columns]
    
    df_reordered = df[np.hstack([list(channel_sets.keys()), 'SequencePosition','Misc'])]    
    return df_reordered

def assemble_condition_df(condition_ind, align, analysisWin, channel_index, mat_contents,tx_name='binnedTX'):
    """Function to collect all repititions of responses to a given condition aligned to go cue
    returns tidy dataframes

    Parameters
    ----------
    condition_ind : int
        condition index for pulling out specific condition. What the condition
        is is experiment depent e.g. int could point to a phoneme or word.
    align : str 
        how to align neural data from trials. must be either 'go' or 'delay' 
    analysisWin : array of int, len(2)
        number of bins to pull out per trial relative to the align cue. length two
        so [start, stop] and integers can be negative meaning before align cue. E.g. [-100, 150] would mean using 100 bins before and  150 bins after the align cue for each trial
    channel_index : array_like
        which recording channels to use. For T12:
        
        * 1-64: superior 6v
        * 65-128: inferior 6v
        * 129-192: inferior 44
        * 193-256: superior 44
    mat_contents : dict
        .mat file contents of data collected with an expected structure. See
        experiment day readme.txt for descriptions of fields

    Returns
    -------
    ts_df : pd.DataFrame (n_trials*time_length, len(channel_index)+3)
        long form (ish) timeseries for each trial of condition with channels as columns 
        plus columns for relative time, cue (i.e. condition), an trial index

    sum_df : pd.DataFrame (n_trials, len(channel_index)+1)
        long form (ish) df of summed spike counts with channels as columns plus a column with the cue(i.e. condition) index.
    """
    # pull out trial indices to include in time binned data frame for tracking individual trials

    trial_idxs = np.where(mat_contents['trialCues']==condition_ind)[0]
    
    if align.lower() not in ['go','delay']:
        raise ValueError(f"`align` value invalid. must be either 'go' or 'delay'. Got {align}")

    trialAlignEpochTime = mat_contents[align.lower() + 'TrialEpochs'][trial_idxs,:]
    n_trials = trialAlignEpochTime.shape[0]
    condition_channel_array = np.zeros((n_trials,                        # n_trials
                                        analysisWin[1] - analysisWin[0], # n_time_bins
                                        len(channel_index))              # n_channels
                                      )
    relative_time_ind_vec = np.arange(analysisWin[0],analysisWin[1])
    for trial in range(n_trials):
        time_ind_vec = relative_time_ind_vec + trialAlignEpochTime[trial,0] - 1 # subtract 1 to convert between matlab and python indexing 
        binned_tx_trial = mat_contents[tx_name][time_ind_vec[:,None],channel_index[None,:]]
        condition_channel_array[trial,:,:] = binned_tx_trial 
    summed_condition_channel_rate = np.sum(condition_channel_array,axis=1)


    # format dataframes
    ch_cols = ['ch'+str(i) for i in channel_index]

    # format 2d spike sums (trials,channels)
    sum_df = pd.DataFrame(summed_condition_channel_rate,columns=ch_cols)
    sum_df['cue'] = condition_ind
    sum_df['trial'] = trial_idxs

    # format 3d time series as 2d (trials,channels) with column for time bin
    ts_bin_dfs = []
    for bin_idx in range(condition_channel_array.shape[1]):
        ts_bin_df = pd.DataFrame(condition_channel_array[:,bin_idx,:], columns = ch_cols)
        ts_bin_df['cue'] = condition_ind
        ts_bin_df['relative_time'] = bin_idx
        ts_bin_df['trial'] = trial_idxs
        ts_bin_dfs.append(ts_bin_df)
    ts_df = pd.concat(ts_bin_dfs)

    return ts_df, sum_df

def assemble_all_conditions(mat_contents,metadata,align,analysisWin,channel_index,tx_name='binnedTX'):
    """Assemble neural data for all experimental conditions with metadata.

    This function processes neural data from all experimental conditions, aligns
    them to specified temporal events, and combines them with trial metadata.

    Parameters
    ----------
    mat_contents : dict
        Loaded .mat file contents with binned neural data and trial timing 
        information from preprocessing. Expected fields include 'trialCues',
        timing epoch fields (e.g., 'goTrialEpochs'), and neural feature arrays.
    metadata : pd.DataFrame
        Trial metadata DataFrame with a 'cue_id' column for linking to neural
        data. Contains experimental conditions and trial information.
    align : str
        Temporal alignment reference, either "go" or "delay" for aligning data
        assembly to go cue or delay period onset.
    analysisWin : array-like, length 2
        Time window in bins relative to alignment cue [start, end]. Negative
        values indicate time before alignment. Example: [-100, 150] uses 100
        bins before and 150 bins after the alignment cue.
    channel_index : array-like
        Neural channel indices to include in the analysis. Allows selection
        of specific recording sites or brain regions.
    tx_name : str, optional
        Name of the neural feature matrix in mat_contents (default: 'binnedTX').
        Common options include threshold crossings or spike power features.

    Returns
    -------
    ts_df : pd.DataFrame
        Time series DataFrame with neural data for all conditions. Contains
        'relative_time' column indicating bins relative to alignment, plus
        all metadata fields for each trial.
    sum_df : pd.DataFrame
        Aggregated DataFrame with summed neural activity over the entire
        analysis window for each trial, combined with trial metadata.
        
    Notes
    -----
    - Uses assemble_condition_df() internally for each unique condition
    - Metadata is merged based on matching cue_id values
    - All neural channels are labeled as 'ch{index}' in output DataFrames
    """
    ts_dfs = []
    sum_dfs = []
    metadata = metadata.copy().set_index('cue_id')
    for cue in np.unique(mat_contents['trialCues']):
        ts_cue, sum_cue = assemble_condition_df(cue,align,analysisWin,channel_index,mat_contents,tx_name=tx_name)
        
        cue_metadata = metadata.loc[cue]
        for metadata_field in cue_metadata.index:
            ts_cue[metadata_field] = cue_metadata.loc[metadata_field]
            sum_cue[metadata_field] = cue_metadata.loc[metadata_field]
        
        ts_dfs.append(ts_cue)
        sum_dfs.append(sum_cue)

    return pd.concat(ts_dfs), pd.concat(sum_dfs)

def assembleGoAlignedWin(fp_mat, conditionFunc, analysisWin, 
                     channel_index = ALL_CH, featureNames=['binnedTX', 'spikePow'], featureColNames = ['tx', 'sp'],
                     zscore=True):
    """Assemble go-aligned neural data with multiple features for analysis.
    
    This function loads neural data from a .mat file, processes multiple neural
    features (e.g., threshold crossings and spike power), and assembles them into
    a unified DataFrame aligned to the go cue for downstream analysis.

    Parameters
    ----------
    fp_mat : str
        Filepath to preprocessed .mat file containing neural data and trial timing.
    conditionFunc : callable
        Function that processes the 'cueList' field from the .mat file and returns
        a DataFrame of trial metadata (e.g., phoneme sequences, movement directions).
        Must return (cues_df, position_columns).
    analysisWin : array-like, length 2
        Time window in bins relative to go cue [start, end]. Negative values
        indicate time before go cue. Example: [-100, 150] for 100 bins before
        to 150 bins after go cue.
    channel_index : array-like, optional
        Neural channel indices to include (default: ALL_CH for all 256 channels).
    featureNames : list of str, optional
        Names of neural feature matrices in .mat file (default: ['binnedTX', 'spikePow']).
    featureColNames : list of str, optional
        Short names for features in output column names (default: ['tx', 'sp']).
    zscore : bool, optional
        Whether to z-score normalize all neural features (default: True).

    Returns
    -------
    delayWin : pd.DataFrame
        DataFrame with neural features and trial metadata. Neural data columns
        are named as 'ch{channel}_{feature}' (e.g., 'ch0_tx', 'ch0_sp').
    channel_map : dict
        Mapping of brain region names to corresponding feature column names
        for region-specific analyses. Built from 'chanSetNames' and 'chanSets'
        fields in the .mat file.
        
    Notes
    -----
    - Automatically converts MATLAB 1-based indexing to Python 0-based indexing
    - Combines multiple neural features into a single analysis-ready DataFrame
    - Provides region-specific channel mapping for anatomical analyses
    """
    # TODO: docstring for all
    dat = loadmat(fp_mat, squeeze_me=True)
    dat['trialCues'] = np.array(dat['trialCues'])-1


    cues = conditionFunc(dat['cueList'])
    for feature in featureNames:
        dat[feature] = dat[feature].astype(float) # to allow for mean subtraction
    
    all_features = []
    for feature, col_name in zip(featureNames, featureColNames):
        _, delayWin = assemble_all_conditions(dat,cues,'go',
                                        analysisWin,
                                        channel_index = channel_index, tx_name=feature)
        neural_channels = [col for col in delayWin.columns if col.startswith('ch')]
        neural_data = delayWin[neural_channels].copy()
        neural_data.columns = [col +f'_{col_name}' for col in neural_data.columns] # add feature name to channel names
        all_features.append(neural_data)
    metadata = delayWin.drop(neural_channels,axis=1).copy() # same for both iterations
    all_features.append(metadata)
    delayWin = pd.concat(all_features,axis=1)
    
    # build channel map for future per-area analysis
    channel_map = {}
    for area, channels in zip(dat['chanSetNames'], dat['chanSets']):
        channels_python = channels - 1 # convert ch indices to python
        feature_cols = []
        for feature_col in featureColNames:
            feature_cols.append([f'ch{ch}_{feature_col}' for ch in channels_python])
        channel_map[area.strip()] = np.hstack(feature_cols) # build df column names

    if zscore:
        ch_col = [col for col in delayWin.columns if col.startswith('ch')]
        delayWin.loc[:,ch_col] = (delayWin[ch_col] - delayWin[ch_col].mean())/delayWin[ch_col].std()
    return delayWin, channel_map


def join_chunk(x, chunk_size, join_function = np.sum, axis=0):
    """
    for an array of 20ms tx crossings for some number of channels,
    rewindow by summing spikes over n_win indices.
    
    Parameters
    ----------
    x : np.array (n_20ms_windows, n_channels)
        array of threshold crossings from redisMat for
        arbitrary channel number
    chunk_size : int 
        number of windows to sum together. E.g. if
        n_win=5 then the result is 5*20=100ms windows or
        the number of tx over 100ms for that channel
    axis : int, optional
        should be 0 to chunk over time, not channels
    """
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return join_function(x,axis=axis+1)

def assemble_trials_metadata_T16(fp_mat,channel_sets,cue_condition_func,analysisWin,window_length_ms=100, bin_time_ms=20,
                             feature_mat_name=['binnedTX', 'spikePow'], feat_join_function = [np.sum, np.mean], feature_column_name = ['tx', 'sp'],
                             zscore=True):
    """assemble data as formatted from T16

    Parameters
    ----------
    fp_mat : str
        filepath to preprocessed .mat file with whole session data
    channel_sets : dict
        names of array regions as keys and associated indices of channels of that array as values
    cue_condition_func : executable
        function that can be called with input of 'cueList' field of .mat
        and will return a dataframe of metadata for each cue like phoneme
        in position of sequence type
    analysisWin : list length 2
        window of number of bins to use before (negative) or after (positive) go cue
        for example, if a trial starts 2 seconds before the go cue and lasts 3 seconds
        after the co gue analysisWin=[-100, 150]
    window_length_ms : int, optional
        number of miliseconds for final window sizes, must be a multiple of bin_time_ms, by default 100ms
    bin_time_ms : int, optional
        number of miliseconds that binned data is in in the .mat file at fp_mat, by default 20ms
    feature_mat_name : list, optional
        list of features to include in final dataframe, by default ['tx_blkMeanSub', 'sp_blkMeanSub']
    feat_join_function : list, optional
        function to aggregate windows over, by default [np.sum, np.mean]
    feature_column_name : list, optional
        feature name that will appear in final dataframe, by default ['tx', 'sp']
    zscore : bool, optional
        whether to zscore the data for all areas, by default True
    Returns
    -------
    dataframe of data from all trials with metadata about the cue for that trial
    """
    if window_length_ms % bin_time_ms != 0:
        raise ValueError(f"window length ({window_length_ms}) incompatible with mat file bin size ({bin_time_ms}), must be a multiple of bin size.")
    window_length_bins = window_length_ms // bin_time_ms

    mat_contents = loadmat(fp_mat, squeeze_me=True)
    # convert to python indexing
    mat_contents['trialCues'] = np.array(mat_contents['trialCues']) -1 # correct for python based indexing
    mat_contents['goTrialEpochs'] = mat_contents['goTrialEpochs'] - 1
    for feature_name in feature_mat_name:
        mat_contents[feature_name] = mat_contents[feature_name].astype(float) # to allow for mean subtraction

    cues, position_columns = cue_condition_func(mat_contents['cueList'])

    trial_data = []
    for trial in range(len(mat_contents['trialCues'])):
        # get delay and go tx in redis 20ms windowing
        trial_go_cue = mat_contents['goTrialEpochs'][trial, 0]

        # parse_metadata
        cue_id = mat_contents['trialCues'][trial]
        trial_mtdata = cues.loc[cue_id].copy()
        trial_mtdata['trial'] = trial

        feature_dfs = []
        for feat, col_identifier, join_fcn in zip(feature_mat_name, feature_column_name, feat_join_function):
            feat20ms = mat_contents[feat][trial_go_cue+analysisWin[0]:trial_go_cue+analysisWin[1]]
            total_expected_bins = analysisWin[1]-analysisWin[0]
            if total_expected_bins != len(feat20ms):
                print(f"mismatch in expected and actual trial duration for trial {trial}. Expected {total_expected_bins} bins, got {len(feat20ms)} bins")
            # make sure delay chunk has number of bins dividible by window length
            # or if not round down by removing from the beginning for delay, end for go
            extraBins = feat20ms.shape[0] % window_length_bins
            if extraBins:
                feat20ms = feat20ms[:(-1*extraBins),:] #shave off extra bins from end of go 
            featTrial = join_chunk(feat20ms,window_length_bins, join_function=join_fcn, axis=0)
        
            feature_trial_df = pd.DataFrame(featTrial,columns = [f'ch{str(i)}_{col_identifier}' for i in range(featTrial.shape[1])]).astype('float32')

            # assume that all delay bins are included bu some go period bins may have been dropped
            # also assume analysisWin[0]<=0
            trialStartTime = analysisWin[0]*bin_time_ms/1000
            trialEndTime = (len(feat20ms) + analysisWin[0])*bin_time_ms/1000

            feature_dfs.append(feature_trial_df)
        
        trial_df = pd.concat(feature_dfs,axis=1)
        trial_df['time(s)'] = np.round(np.linspace(trialStartTime, trialEndTime-window_length_bins*bin_time_ms/1000, len(feature_trial_df)),3)

            
        trial_metadata_df = trial_df.merge(pd.DataFrame([trial_mtdata for _ in range(len(trial_df))]).reset_index(drop=True),
                                        left_index=True, right_index=True)
        trial_data.append(trial_metadata_df)

    formated_data = add_columns_groups(pd.concat(trial_data,ignore_index=True),channel_sets, position_columns)
    if zscore:
        areas = list(channel_sets.keys())
        multiindexed_columns = [col for col in formated_data.columns if col[0] in areas]
        formated_data.loc[:,multiindexed_columns] = (formated_data.loc[:,multiindexed_columns] - formated_data.loc[:,multiindexed_columns].mean())/formated_data.loc[:,multiindexed_columns].std()
        
    return formated_data

##########     cue condition functions     ##########
# to parse individual experiment cue conditions into movments at each position
def cueConditionsSingle(cueList):
    position_columns = ['First']
    cues = pd.DataFrame(cueList,columns=position_columns)
    # cues.loc[np.where(cues[position_columns]=='Do Nothing')[0][0],position_columns]=np.nan
    cues['cue_id'] = cues.index.values.squeeze()
    cues['cue'] = cueList
    return cues

def cueConditionsSeq(cueList):
    position_columns = ['First','Second','Third']
    elements = [np.hstack([i,c.split(' ')]) for i, c in enumerate(cueList) if c!= 'DO_NOTHING']
    cues = pd.DataFrame(elements,columns = ['index','First','Second','Third'])
    cues['index'] = cues['index'].astype(float)
    cues.set_index('index',inplace=True)
    if 'Do Nothing' in cueList:
        cues.loc[np.where(np.array(cueList)=='Do Nothing')[0][0]]=np.nan
    elif 'DO_NOTHING' in cueList:
        cues.loc[np.where(np.array(cueList)=='DO_NOTHING')[0][0]]=np.nan
    cues.sort_index(inplace=True)
    cues['cue_id'] = cues.index
    cues['cue'] = cueList
    return cues

def cueConditionsMentalStrategySwap(cueList):
    mentalStrategyAll = []
    pos0All = []
    pos1All = []
    pos2All = []

    for cue in cueList:
        if cue == 'DO_NOTHING':
            pos0 = pos1 = pos2 = float('nan')
            mentalStrategy = 'DO_NOTHING'
        else:
            mentalStrategy, sequence = cue.split('-')
            pos0, pos1, pos2 = sequence.split(' ')
        
        
        
        mentalStrategyAll.append(mentalStrategy)
        pos0All.append(pos0)
        pos1All.append(pos1)
        pos2All.append(pos2)
    
    cueConditions = pd.DataFrame({'mentalStrategy':mentalStrategyAll,
                                  'First': pos0All,
                                  'Second':pos1All,
                                  'Third':pos2All})
    cueConditions['cue'] = cueList
    cueConditions['cue_id'] = np.arange(len(cueList))
    return cueConditions

def cueConditionsSpeak(cueList):
    seqPositionColumns = ['First', 'Second','Third']
    arrowParsingDict = {'up':'↑',
                        'right':'→',
                        'DO_NOTHING':None}

    cueConditions = pd.DataFrame([[arrowParsingDict[arrow] for arrow in c.split('_')] for c in cueList],
                                    columns = seqPositionColumns)
    cueConditions['cue'] = cueList
    cueConditions['cue_id'] = np.arange(len(cueList))
    return cueConditions

def cueConditionsInstructedInnerspeech(cueList):
    seqPositionColumns = ['First', 'Second','Third']
    cueConditions = pd.DataFrame([c.split(' ') for c in cueList],
                                 columns = seqPositionColumns)
    cueConditions['cue'] = cueList
    cueConditions.loc[cueConditions.cue=='DO_NOTHING',seqPositionColumns] = [np.nan]*len(seqPositionColumns)
    cueConditions['cue_id'] = np.arange(len(cueList))
    return cueConditions

def cueConditionsLines(cueList):
    seqPositionColumns = ['First', 'Second','Third']
    cueConditions = pd.DataFrame([c.split('.png')[0].split('_') for c in cueList],
                                 columns = seqPositionColumns)
    cueConditions['cue'] = cueList
    cueConditions['cue_id'] = np.arange(len(cueList))
    return cueConditions,seqPositionColumns

def cueConditionsArrowSeq_T16(cueList):
    position_columns = ['First','Second','Third']
    elements = [np.hstack([i,c.split(' ')]) for i, c in enumerate(cueList) if c!= 'DO_NOTHING']
    cues = pd.DataFrame(elements,columns = ['index','First','Second','Third'])
    cues['index'] = cues['index'].astype(float)
    cues.set_index('index',inplace=True)
    cues['cue_id'] = cues.index
    cues['cue'] = cueList
    return cues, position_columns

def cueConditionsUninstructedLines_T16(cueList):
    if type(cueList[0]) is np.ndarray:
        cueList = [c[0] for c in cueList.flatten()]
    position_columns = ['First','Second','Third']
    elements = [np.hstack([i,c.split(' ')]) for i, c in enumerate(cueList) if c!= 'DO_NOTHING']
    cues = pd.DataFrame(elements,columns = ['index','First','Second','Third'])
    cues['index'] = cues['index'].astype(float)
    cues.set_index('index',inplace=True)
    cues['cue_id'] = cues.index
    cues['cue'] = cueList

    return cues, position_columns

def cueConditionsSpeakingT16(cueList):
    position_columns = ['First']
    cues = pd.DataFrame(cueList, columns = position_columns)
    cues.loc[:,position_columns] = cues.loc[:,position_columns].replace({'"RIGHT"   ':'→', '"UP"      ':'↑'})
    cues['cue'] = cueList
    cues.loc[cues.cue=='Do Nothing', position_columns] = np.nan
    return cues, position_columns
