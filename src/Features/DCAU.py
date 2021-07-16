from .DE import differential_entropy
import numpy as np

def differential_caudality(signal, channel_name):
    """
    Calculate the differential caudality of EEG singnal
    The differential entropy of the posterior brain channel was subtracts from that of the corresponding frontal brain channel
    
    input:
        signal: numpy array. The temporal signal (>=2d) for feature extraction.
        channel_name: list or 1d array. The names of all the channels of the EEG signal
    return
        dcau: numpy array. 
    """
    if len(channel_name) != signal.shape[-2]:
        raise ValueError("The length of channel_name must be the same as the number of channels of the signal")
    DE = differential_entropy(signal)
    
    def cal_dcau(de,cn):
        signal_DE = dict(zip(cn, de))
        y = []
        for key in signal_DE:
            if key == 'Nz':
                y.append(signal_DE['Nz'] - signal_DE['Iz'])
            if key == 'Fp1':
                y.append(signal_DE['Fp1'] - signal_DE['O1'])
            if key == 'Fpz':
                y.append(signal_DE['Fpz'] - signal_DE['Oz'])
            if key == 'Fp2':
                y.append(signal_DE['Fp2'] - signal_DE['O2'])
            if key == 'AF7':
                y.append(signal_DE['AF7'] - signal_DE['PO7'])
            if key == 'AF3':
                y.append(signal_DE['AF3'] - signal_DE['PO3'])
            if key == 'AFz':
                y.append(signal_DE['AFz'] - signal_DE['POz'])
            if key == 'AF4':
                y.append(signal_DE['AF4'] - signal_DE['PO4'])
            if key == 'AF8':
                y.append(signal_DE['AF8'] - signal_DE['PO8'])
            if key == 'F9':
                y.append(signal_DE['F9'] - signal_DE['P9'])
            if key == 'F7':
                y.append(signal_DE['F7'] - signal_DE['P7'])
            if key == 'F5':
                y.append(signal_DE['F5'] - signal_DE['P5'])
            if key == 'F3':
                y.append(signal_DE['F3'] - signal_DE['P3'])
            if key == 'F1':
                y.append(signal_DE['F1'] - signal_DE['P1'])
            if key == 'Fz':
                y.append(signal_DE['Fz'] - signal_DE['Pz'])
            if key == 'F2':
                y.append(signal_DE['F2'] - signal_DE['P2'])
            if key == 'F4':
                y.append(signal_DE['F4'] - signal_DE['P4'])
            if key == 'F6':
                y.append(signal_DE['F6'] - signal_DE['P6'])
            if key == 'F8':
                y.append(signal_DE['F8'] - signal_DE['P8'])
            if key == 'F10':
                y.append(signal_DE['F10'] - signal_DE['P10'])
            if key == 'FT9':
                y.append(signal_DE['FT9'] - signal_DE['TP9'])
            if key == 'FT7':
                y.append(signal_DE['FT7'] - signal_DE['TP7'])
            if key == 'FC5':
                y.append(signal_DE['FC5'] - signal_DE['CP5'])
            if key == 'FC3':
                y.append(signal_DE['FC3'] - signal_DE['CP3'])
            if key == 'FC1':
                y.append(signal_DE['FC1'] - signal_DE['CP1'])
            if key == 'FCz':
                y.append(signal_DE['FCz'] - signal_DE['CPz'])
            if key == 'FC2':
                y.append(signal_DE['FC2'] - signal_DE['CP2'])
            if key == 'FC4':
                y.append(signal_DE['FC4'] - signal_DE['CP4'])
            if key == 'FC6':
                y.append(signal_DE['FC6'] - signal_DE['CP6'])
            if key == 'FT8':
                y.append(signal_DE['FT8'] - signal_DE['TP8'])
            if key == 'FT10':
                y.append(signal_DE['FT10'] - signal_DE['TP10'])
        return np.array(y)
    
    if len(DE.shape) == 1:
        return cal_dcau(DE, channel_name)
    else:
        return np.apply_along_axis(cal_dcau, axis = -1, arr = DE, cn = channel_name)
