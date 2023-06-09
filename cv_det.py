import cv2
import os
import numpy as np
import pandas as pd
import time
from progress.bar import Bar

def find_defects(path_to_img, path_to_resoult, save_images: bool):
    img = cv2.imread(path_to_img)

    edged = cv2.Canny(img, 100, 255)

    edges, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if save_images:
        img = cv2.drawContours(img, contours=edges, contourIdx=-1, color=(0, 0, 255),  thickness=1)
        cv2.imwrite(path_to_resoult, img)

    return len(edges)//2



def prepare_res_dir(path):
    dirs = np.array(os.listdir(path)[:-4])
    
    #chiking if dir is empty
    if len(os.listdir('resoult/')) == 0:
        for dir in dirs:
            os.mkdir(f'resoult/{dir}')
    else: pass


def get_files_list(path):

    #getting dirs list but removing .TIF and ADC files
    dirs = np.array(os.listdir(path)[:-4])


    files_list = np.array([])

    for dir in dirs:
        for file in os.listdir(f'{path}{dir}'):
            if file!='Thumbs.db':
                files_list = np.append(files_list, [f'{dir}/{file}'])

    return files_list



def detect_wafer(path):

    #getting dirs list but removing .TIF files
    files = get_files_list(path)

    #progress bar
    bar = Bar('Detecting', max=len(files))

    #preparing dirs for resoults
    prepare_res_dir(path)

    defects = np.array([])
    for file in files:
        defects = np.append(defects, find_defects(f'{path}{file}', f'result/{file}', save_images=False))
        bar.next()

    #making DataFrame
    files = np.reshape(files, (files.shape[0], 1))
    defects = np.reshape(defects, (defects.shape[0], 1))

    data = np.append(files, defects, axis=1)

    df = pd.DataFrame(data=data, columns=['file_name', 'defects_count'])
    df.to_csv('result.csv', index=False)

    bar.finish()

    return df

def init_wafer_map(path):
    img  = cv2.imread('wafer.jpg')

    #MARKING UP THE MAP
    img = cv2.rectangle(img, (0, 0), (1100, 1100), (0, 0, 0), -1)
    img = cv2.circle(img, (550, 550), 505, (192, 192, 192), -1)

    for i in range(45, 1055, 5):
        for j in range(45, 1055, 40):
            img = cv2.rectangle(img, (i, j), (i+5, j+40), (155, 0, 0), 1)


    cv2.imwrite('wafer.jpg', img)

#deffects_regions should be an array contains col and line number of def region
# [(x1, y1), ..., (xn, yn)]
def markup_defect_regions(defects_regions):
    img  = cv2.imread('wafer.jpg')

    print('dr=', defects_regions[0:5])
    for pos in defects_regions:
        x = (pos[0]-5) * 5 + 40
        y = (pos[1]-1) * 40 + 45
        def_count = pos[2]
        
        if def_count < 3:
            img = cv2.rectangle(img, (x, y), (x+5, y+40), (0, 100, 0), -1)
        elif def_count > 3 and def_count < 10:
            img = cv2.rectangle(img, (x, y), (x+5, y+40), (80, 127, 255), -1)
        elif def_count > 10:
            img = cv2.rectangle(img, (x, y), (x+5, y+40), (0, 0, 255), -1)

    cv2.imwrite('wafer.jpg', img)
