def find_most_frequent_number(x):
    """
    :param x: list
    :return: the most frequent number in the list
    """
    count_dict = {}
    for value in x:
        if value in count_dict:
            count_dict[value] += 1
        else:
            count_dict[value] = 1

    # 找到出现次数最多的元素（众数）
    mode = max(count_dict, key=count_dict.get)
    return mode

def find_amplitude(x):
    from scipy.signal import find_peaks
    import numpy as np
    """
    :param x: concentration
    :return: the amplitude of the concentration
    """
    pos_peaks,_=find_peaks(x)
    neg_peaks,_=find_peaks(-x)
    length=min(len(pos_peaks),len(neg_peaks))
    pos_peaks=pos_peaks[:length]
    neg_peaks=neg_peaks[:length]
    amp=np.zeros(len(pos_peaks))
    for i in range (0,len(pos_peaks)):
        amp[i]=abs(x[pos_peaks[i]]-x[neg_peaks[i]])
    return max(amp)

def find_period(x):
    from scipy.signal import find_peaks
    import numpy as np
    """
    :param x: concentration
    :return: the period of the concentration
    """
    pos_peaks,_=find_peaks(x)
    period=np.zeros(len(pos_peaks)-1)
    for i in range (0,len(pos_peaks)-1):
        period[i]=abs(pos_peaks[i+1]-pos_peaks[i])
    return float(find_most_frequent_number(period))
