"""
Routines for scaling biomedical image data by window level and width.
"""

import numpy as np

win_dict = {'abdomen':
            {'wl': 60, 'ww': 400},
            'angio':
            {'wl': 300, 'ww': 600},
            'bone':
            {'wl': 300, 'ww': 1500},
            'brain':
            {'wl': 40, 'ww': 80},
            'chest':
            {'wl': 40, 'ww': 400},
            'lungs':
            {'wl': -400, 'ww': 1500}}

def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)
        

def ct_win(im, wl, ww, dtype, out_range):    
    """
    Scale CT image represented as a `pydicom.dataset.FileDataset` instance.
    """

    # Convert pixel data from Houndsfield units to intensity:
    intercept = int(im[(0x0028, 0x1052)].value)
    slope = int(im[(0x0028, 0x1053)].value)
    data = (slope*im.pixel_array+intercept)

    # Scale intensity:
    return win_scale(data, wl, ww, dtype, out_range)