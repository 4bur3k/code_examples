import numpy as np
import pandas as pd
import cv2 as cv
import os
import natsort
import logging
import re
import math

logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")

def get_files_list(extention: str, path):
    files = os.listdir(path)

    ase_files_list = list()

    for filename in files:
        if re.fullmatch(r'.+\.'+extention, filename):
            ase_files_list.append(filename)
    
    return ase_files_list


#creating a directory for results
def init_dirs(path):
    try:
        os.mkdir(path+'/res')
        os.mkdir(path+'/res/data')
        os.mkdir(path+'/res/img')
    except FileExistsError as e:
        pass

#input: path to dir with data
#output: dir (./res/) w\ .csv files
def get_parse_data(path):
    aso_files_list = get_files_list('aso', path)

    for i in range(len(aso_files_list)):

        '''
        Parsing info from .ase files
        Resoult CSV contains:
        cluster_pos_x\y - real positioons of clusters
        pixel_pos_x\y - positions to draw
        cluster_size: {0 : 0.16-0.2, 1 : 0.2-0.3, 2 : 0.3-0.36, 3 : Area}
        '''

        init_dirs(path)

        with open(path+'/'+aso_files_list[i], 'rt', encoding='utf-16') as aso_f:
            aso_data = aso_f.read()

        clusters = re.findall(r'\nCLUSTER_DESCR.+#', aso_data, flags=re.MULTILINE)
        array = np.array([np.zeros(5)])

        for string in clusters:
            match = re.findall(r'.+?;', string)
            cluster_size = match[4][:-1]
            cluster_pos_x = float(match[8][:-1])
            cluster_pos_y = float(match[9][:-1])
            pixel_pos_x = float(match[13][:-1])
            pixel_pos_y = float(match[14][:-1])

            
            array = np.append(array, [[cluster_pos_x, cluster_pos_y, pixel_pos_x, pixel_pos_y, cluster_size]], axis=0)

            
        array = array[1:]

        df = pd.DataFrame(data=array, columns=['c_pos_x', 'c_pos_y', 'pixel_pos_x', 'pixel_pos_y', 'c_size'])

        df.to_csv(f'{path}/res/data/{i}_data.csv')


def find_overlap(x, y, ds_x, ds_y, radius):
    # distances between given value and values in ds 
    array_of_distances = np.array([])

    for i in range(len(ds_x)):
        x_dist = abs(x - ds_x[i])
        y_dist = abs(y - ds_y[i])
        dist = math.sqrt((x_dist*x_dist + y_dist*y_dist))
        array_of_distances = np.append(array_of_distances, dist)

    if min(array_of_distances) <= radius:
        return True
    
    else: return False
    


def count_res_data(radius, path):

    files_list = natsort.natsorted([f'{path}/res/data/' + i for i in os.listdir(f'{path}/res/data/')])

    res_df = pd.DataFrame(data = None, columns=['0.16-0.2', '0.2-0.3', '0.3-0.36', 'Area', 'Rep', 'New', 'All'])

    for i in range(len(files_list)):

        overlaps_counter = 0
        #small, medium, large, area clusters
        s_counter = 0
        m_counter = 0
        l_counter = 0
        a_counter = 0

        if i != 0:
            current_data = pd.read_csv(files_list[i])
            prev_data = pd.read_csv(files_list[i-1])
        else:
            current_data = pd.read_csv(files_list[i])
            prev_data = pd.DataFrame()
        
        n = len(current_data)
        
    
        green = list()
        if prev_data.empty == False:
            for j in range(n):
                #cheking out if clusters on the same positions
                is_overlapd = find_overlap(current_data['c_pos_x'][j], current_data['c_pos_y'][j], 
                                        prev_data['c_pos_x'], prev_data['c_pos_y'], radius)
                # print(j, overlaps_counter)
                if is_overlapd:
                    overlaps_counter += 1
                    green.append(j)
                    

        s_counter = len(current_data[current_data['c_size'] == '0.16 - 0.20'])
        m_counter = len(current_data[current_data['c_size'] == '0.20 - 0.30'])
        l_counter = len(current_data[current_data['c_size'] == '0.30 - 0.36'])
        a_counter = len(current_data[current_data['c_size'] == 'Area'])

        new_clusters = len(current_data) - overlaps_counter


        data = [s_counter, m_counter, l_counter, a_counter, overlaps_counter, new_clusters, len(current_data)]
        res_df.loc[len(res_df.index)] = data

        if prev_data.empty == False:
            draw_defects(current_data['pixel_pos_x'], current_data['pixel_pos_y'], prev_data['pixel_pos_x'], 
                        prev_data['pixel_pos_y'], green, i, path)
        else:
            draw_defects(current_data['pixel_pos_x'], current_data['pixel_pos_y'], pd.DataFrame([]), 
                        pd.DataFrame([]), green, i, path)
        
        draw_orig_defects(current_data['pixel_pos_x'], current_data['pixel_pos_y'], current_data['c_size'], i, path)

    res_df.to_csv(f'{path}/res/res.csv')

def draw_defects(x_array, y_array, x_prev_array, y_prev_array, arr, index, path):
    img = cv.imread('./system/base2.png')
    # print('here', index)

    n_x = 15
    n_y = 10

    for i in range(len(x_array)):
        if i in arr:
            img = cv.rectangle(img, (int(x_array[i]/n_x), int(y_array[i]/n_y)), (int(x_array[i]/n_x)+10, int(y_array[i]/n_y)+10), (0, 0, 255), -1)
        else:
            img = cv.rectangle(img, (int(x_array[i]/n_x), int(y_array[i]/n_y)), (int(x_array[i]/n_x)+10, int(y_array[i]/n_y)+10), (0, 255, 0), -1)
    
    for i in range(len(x_prev_array)):
        img = cv.rectangle(img, (int(x_prev_array[i]/n_x), int(y_prev_array[i]/n_y)), (int(x_prev_array[i]/n_x)+10, int(y_prev_array[i]/n_y)+10), (255, 0, 0), 1)

    cv.imwrite(f'{path}/res/img/{index}.png', img)

def draw_orig_defects(x_array, y_array, size_array, index, path):
    img = cv.imread('./system/base2.png')

    n_x = 15
    n_y = 10

    color = (0, 0, 0)

    for i in range(len(x_array)):
        if size_array[i] == '0.16 - 0.20':
            color = (0, 255, 0)
        elif size_array[i]  == '0.20 - 0.30':
            color = (0, 255, 255)
        elif size_array[i]  == '0.30 - 0.36':
            color = (255, 0, 0)
        elif size_array[i]  == 'Area':
            color = (0, 0, 255)
        img = cv.rectangle(img, (int(x_array[i]/n_x), int(y_array[i]/n_y)), (int(x_array[i]/n_x)+10, int(y_array[i]/n_y)+10), color, -1)
    
    cv.imwrite(f'{path}/res/img/{index}_orig.png', img)