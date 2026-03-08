exactColorPalette = {
    (0, 0, 0): 0,       # Background
    (255, 0, 0): 1,     # Class 1
    (0, 255, 0): 2,     # Class 2
    (0, 0, 255): 3,     # Class 3
    (255, 255, 0): 4,   # Class 4
    (255, 0, 255): 5,   # Class 5
    (0, 255, 255): 6,   # Class 6
}

rangeColorPalette = {
    "class7": {"range": ((183,189), (0,0), (250,255)), "classIndex": 7}, # Class 7 needs to be range
    "class8": {"range": ((200,255), (113,192), (0,0)), "classIndex": 8}, # Class 8 needs to be range
    "class9": {"range": ((0,0), (221,255), (162,189)), "classIndex": 9} # Class 9 needs to be range
}

inferenceColorPalette = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Class 1
    2: (0, 255, 0),     # Class 2
    3: (0, 0, 255),     # Class 3
    4: (255, 255, 0),   # Class 4
    5: (255, 0, 255),   # Class 5
    6: (0, 255, 255),   # Class 6
    7: (186, 0, 255),   # Class 7
    8: (255, 113, 0),   # Class 8
    9: (0, 221, 162),   # Class 9
}
