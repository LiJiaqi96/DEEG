from .DE import differential_entropy
import numpy as np

def differential_asymmetry(signal, channel_name):
    """
    Calculate the differential asymmetry of EEG singnal
    The differential entropy of the right brain channel was subtracts from that of the corresponding left brain channel
    
    input:
        signal: numpy array. The temporal signal (>=2d) for feature extraction
        channel_name: list or 1d array. The names of all the channels of the EEG signal
    return
        dasm: numpy array. 
    """
    if len(channel_name) != signal.shape[-2]:
        raise ValueError("The length of channel_name must be the same as the number of channels of the signal")
        
    DE = differential_entropy(signal)
    
    def cal_dasm(de,cn):
        signal_DE = dict(zip(cn, de))
        y = []
        for key in signal_DE:
            if key == 'Fp1':
                y.append(signal_DE['Fp1'] - signal_DE['Fp2'])
            if key == 'AF7':
                y.append(signal_DE['AF7'] - signal_DE['AF8'])
            if key == 'AF3':
                y.append(signal_DE['AF3'] - signal_DE['AF4'])
            if key == 'F9':
                y.append(signal_DE['F9'] - signal_DE['F10'])
            if key == 'F7':
                y.append(signal_DE['F7'] - signal_DE['F8'])
            if key == 'F5':
                y.append(signal_DE['F5'] - signal_DE['F6'])
            if key == 'F3':
                y.append(signal_DE['F3'] - signal_DE['F4'])
            if key == 'F1':
                y.append(signal_DE['F1'] - signal_DE['F2'])
            if key == 'FT9':
                y.append(signal_DE['FT9'] - signal_DE['FT10'])
            if key == 'FT7':
                y.append(signal_DE['FT7'] - signal_DE['FT8'])
            if key == 'FC5':
                y.append(signal_DE['FC5'] - signal_DE['FC6'])
            if key == 'FC3':
                y.append(signal_DE['FC3'] - signal_DE['FC4'])
            if key == 'FC1':
                y.append(signal_DE['FC1'] - signal_DE['FC2'])
            if key == 'LPA':
                y.append(signal_DE['LPA'] - signal_DE['RPA'])
            if key == 'T7':
                y.append(signal_DE['T7'] - signal_DE['T8'])
            if key == 'C5':
                y.append(signal_DE['C5'] - signal_DE['C6'])
            if key == 'C3':
                y.append(signal_DE['C3'] - signal_DE['C4'])
            if key == 'C1':
                y.append(signal_DE['C1'] - signal_DE['C2'])
            if key == 'TP9':
                y.append(signal_DE['TP9'] - signal_DE['TP10'])
            if key == 'TP7':
                y.append(signal_DE['TP7'] - signal_DE['TP8'])
            if key == 'CP5':
                y.append(signal_DE['CP5'] - signal_DE['CP6'])
            if key == 'CP3':
                y.append(signal_DE['CP3'] - signal_DE['CP4'])
            if key == 'CP1':
                y.append(signal_DE['CP1'] - signal_DE['CP2'])
            if key == 'P9':
                y.append(signal_DE['P9'] - signal_DE['P10'])
            if key == 'P7':
                y.append(signal_DE['P7'] - signal_DE['P8'])
            if key == 'P5':
                y.append(signal_DE['P5'] - signal_DE['P6'])
            if key == 'P3':
                y.append(signal_DE['P3'] - signal_DE['P4'])
            if key == 'P1':
                y.append(signal_DE['P1'] - signal_DE['P2'])
            if key == 'PO7':
                y.append(signal_DE['PO7'] - signal_DE['PO8'])
            if key == 'PO3':
                y.append(signal_DE['PO3'] - signal_DE['PO4'])
            if key == 'O1':
                y.append(signal_DE['O1'] - signal_DE['O2'])
        return np.array(y)
    
    if len(DE.shape) == 1:
        return cal_dasm(DE, channel_name)
    else:
        return np.apply_along_axis(cal_dasm, axis=-1, arr=DE, cn=channel_name)
