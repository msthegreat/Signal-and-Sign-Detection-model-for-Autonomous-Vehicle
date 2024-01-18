import numpy as np


sign_labels_count = {
    'stop': 0,
    'signalAhead': 0,
    'pedestrianCrossing': 0,
}

light_labels_count = {
    'stop': 0,
    'go': 0
}

sign_labels = {
    1: 'pedestrianCrossing',
    2: 'signalAhead',
    3: 'stop',
    0: 'unknown'
}

light_labels = {
    1: 'go',
    2: 'stop',
    0: 'unknown'
}

signs = {
    1: 'yellow',
    2: 'yellow',
    3: 'red-upper',
}

lights = {
    1: 'green',
    2: 'red',
}

text = {
    'stop': 'stop',
    'pedestrianCrossing': 'pCrossing',
    'signalAhead': 'sAhead'
}

COLOR_THRESHOLDS = {
    'yellow': np.array([np.array([16, 100, 95]), np.array([30, 230, 200])]),
    'red-upper': np.array([np.array([155, 140, 150]), np.array([179, 255, 255])]),
    'red-lower': np.array([np.array([0, 0, 0]), np.array([18, 80, 130])])
}

LIGHT_THRESHOLDS = {
    'red': [np.array([np.array([160, 120, 140]), np.array([179, 255, 255])]), \
                    np.array([np.array([0, 150, 150]), np.array([12, 255, 255])])],
     'green': np.array([np.array([40, 120, 110]), np.array([90, 225, 255])])

}