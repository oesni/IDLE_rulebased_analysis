import os
import sys
import parse_xml
import copy
import numpy as np
import cv2
import json
import glob
from collections import OrderedDict


def calculateMaxSubmission(y, x, csv_arr):
    submission_arr = []
    if x > 0 and y > 0:
        submission_arr.append(csv_arr[y][x]-csv_arr[y-1][x-1])
    if y > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x])
    if x < len(csv_arr[0])-1 and y>0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x+1])
    if x > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x-1])
    if x < len(csv_arr[0])-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x+1])
    if x > 0 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x-1])
    if y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x])
    if x < len(csv_arr[0])-1 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x+1])

    return max(submission_arr)

def writeJason(xmin, ymin, xmax, ymax, tmin, tmax, tmean, object_class, hp_contour, rp_contour):
    object_data = OrderedDict()
    object_data["xmin"] = xmin
    object_data["ymin"] = ymin
    object_data["xmax"] = xmax
    object_data["ymax"] = ymax
    object_data["tmin"] = tmin
    object_data["tmax"] = tmax
    object_data["tmean"] = tmean
    object_data["class"] = object_class
    object_data["hp"] = []

    for i in range(len(hp_contour)):
        temp_contour = []
        for j in range(len(hp_contour[i])):
            temp_contour.append('('+str(hp_contour[i][j][0][0]+xmin)+','+str(hp_contour[i][j][0][1]+ymin)+')')
        object_data["hp"].append(temp_contour)

    object_data["rp"] = []
    for i in range(len(rp_contour)):
        temp_contour = []
        for j in range(len(rp_contour[i])):
            temp_contour.append(
                '(' + str(rp_contour[i][j][0][0] + xmin) + ',' + str(rp_contour[i][j][0][1] + ymin) + ')')
        object_data["rp"].append(temp_contour)
    return object_data

def writeJson2(input_data):
    # print('write "{0}" ...'.format(inp))

    data = OrderedDict()
    # xmin
    data["xmin"] = input_data["xmin"]
    input_data.pop("xmin", None)
    # ymin
    data["ymin"] = input_data["ymin"]
    input_data.pop("ymin", None)
    # xmax
    data["xmax"] = input_data["xmax"]
    input_data.pop("xmax", None)
    # ymax
    data["ymax"] = input_data["ymax"]
    input_data.pop("ymax", None)
    # tmin
    data["tmin"] = input_data["tmin"]
    input_data.pop("tmin", None)
    # tmax
    data["tmax"] = input_data["tmax"]
    input_data.pop("tmax", None)
    # tmean
    data["tmean"] = input_data["tmean"]
    input_data.pop("tmean", None)
    # class
    data["class"] = input_data["class"]
    input_data.pop("class", None)
    
    # hp
    data["hp"] = []
    hp_contour = input_data["hp_counter"]
    for i in range(len(hp_contour)):
        temp_contour = []
        for j in range(len(hp_contour[i])):
            temp_contour.append('('+str(hp_contour[i][j][0][0]+data["xmin"])+','+str(hp_contour[i][j][0][1]+data["ymin"])+')')
        data["hp"].append(temp_contour)
    input_data.pop("hp_counter", None)

    # rp
    data["rp"] = []
    rp_contour = input_data["rp_counter"]
    for i in range(len(rp_contour)):
        temp_contour = []
        for j in range(len(rp_contour[i])):
            temp_contour.append(
                '(' + str(rp_contour[i][j][0][0] + data["xmin"]) + ',' + str(rp_contour[i][j][0][1] + data["ymin"]) + ')')
        data["rp"].append(temp_contour)
    input_data.pop("rp_counter", None)

    ## Emissivity
    if "Emissivity" in input_data:
        data["Emissivity"] = input_data["Emissivity"]
        input_data.pop("Emissivity", None)
    else:
        data["Emissivity"] = 0.95
    ## Atmospheric Temperature
    if "Atmospheric Temperature" in input_data:
        data["Atmospheric Temperature"] = input_data["Atmospheric Temperature"]
        input_data.pop("Atmospheric Temperature", None)
    else:
        data["Atmospheric Temperature"] = 20
    ## Relative Humidity
    if "Relative Humidity" in input_data:
        data["Relative Humidity"] = input_data["Relative Humidity"]
        input_data.pop("Relative Humidity", None)
    else:
        data["Relative Humidity"] = 40

    ## Point
    if "Point" in input_data:
        data["Point"] = input_data["Point"]
        input_data.pop("Point", None)
    else:
        data["Point"] = "2320-472-M-WV-07PA"

    ## FacilityName
    if "FacilityName" in input_data:
        data["FacilityName"] = input_data["FacilityName"]
        input_data.pop("FacilityName", None)
    else:
        data["FacilityName"] = "Fuse"
    ## FileName
    if "FileName" in input_data:
        data["FileName"] = input_data["FileName"]
        input_data.pop("FileName", None)
    else:
        data["FileName"] = "fuse.jpg"
    ## FacilityClass
    if "FacilityClass" in input_data:
        data["FacilityClass"] = input_data["FacilityClass"]
        input_data.pop("FacilityClass", None)
    else:
        data["FacilityClass"] = "ETC"

    ## FacilityClass_option
    if "FacilityClass_option" in input_data:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + str(input_data["FacilityClass_option"])
        input_data.pop("FacilityClass_option", None)
    else:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + "A상"
    ## Limit Temperature
    if "Limit Temperature" in input_data:
        data["Limit Temperature"] = input_data["Limit Temperature"]
        input_data.pop("Limit Temperature", None)
    else:
        data["Limit Temperature"] = 25
    ## PointTemperature
    if "PointTemperature" in input_data:
        data["PointTemperature"] = input_data["PointTemperature"]
        input_data.pop("PointTemperature", None)
    else:
        data["PointTemperature"] = 35.9
    ## Over temperature
    if "Over temperature" in input_data:
        data["Over temperature"] = input_data["Over temperature"]
        input_data.pop("Over temperature")
    else:
        data["Over temperature"] = "9.9"

    ## Over temperature_option
    if "Over temperature_option" in input_data:
        data["Over temperature_option"] = str(data["Over temperature_option"]) + ", " + str(input_data["Over temperature_option"])
        input_data.pop("Over temperature_option", None)
    else:
        data["Over temperature"] = str(data["Over temperature"]) + ", " + "정상"
    ## Cause of Failure
    if "Cause of Failure" in input_data:
        data["Cause of Failure"] = input_data["Cause of Failure"]
        input_data.pop("Cause of Failure", None)
    else:
        data["Cause of Failure"] = "정상"
    ## DiagnosisCode
    if "DiagnosisCode" in input_data:
        data["DiagnosisCode"] = input_data["DiagnosisCode"]
        input_data.pop("DiagnosisCode", None)
    else:
        data["DiagnosisCode"] = "AA"
    ## Diagnosis
    if "Diagnosis" in input_data:
        data["Diagnosis"] = input_data["Diagnosis"]
        input_data.pop("Diagnosis", None)
    else:
        data["Diagnosis"] = "정상임"
    
    print("unused key:")
    for key in input_data.keys():
        print(key)
    
    return data

def analyze(xml, csv, out_dir):
    #xml = Annotation file

    fname = os.path.basename(os.path.splitext(xml)[0])
    analyze = parse_xml.parcingXml(xml)
    analyze = np.array(analyze)
    csv_arr = parse_xml.parcingCsv(csv)
    file_data = OrderedDict()
    file_data["facilities"] = []

    for i in range(len(analyze[0])):
        xmin = int(analyze[0][i])
        xmax = int(analyze[1][i])
        ymin = int(analyze[2][i])
        ymax = int(analyze[3][i])
        object_class = analyze[4][i]
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        csv_flat = csv_crop.flatten()
        csv_flat = np.round_(csv_flat, 1)
        temp_min = csv_flat.min()
        temp_max = csv_flat.max()
        temp_average = np.average(csv_flat)
        temp_average = np.round_(temp_average, 1)

        # find heating points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0 ,255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        hp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find reflection points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        CRITICAL_GRAD = 0.4
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)

        height, width = thresh_arr.shape
        suspected_points = []
        for i in range(height):
            for j in range(width):
                if thresh_arr[i][j] != 0:
                    temp = calculateMaxSubmission(i, j, csv_crop)
                    if temp > CRITICAL_GRAD:
                        suspected_points.append([j,i])

        masking_img = np.zeros((height, width, 3), dtype=np.uint8)
        for pts in suspected_points:
            xy = np.array(pts)
            cv2.circle(masking_img, (xy[0], xy[1]), 3, (255, 255, 255), -1)

        masking_img = masking_img[:,:,0]
        rp_contour, heirachy = cv2.findContours(masking_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # object_data = writeJason(xmin, ymin, xmax, ymax, temp_min, temp_max, temp_average, object_class, hp_contour, rp_contour)
        json_data = {}
        json_data["xmin"] = xmin
        json_data["ymin"] = ymin
        json_data["xmax"] = xmax
        json_data["ymax"] = ymax
        json_data["tmin"] = temp_min
        json_data["tmax"] = temp_max
        json_data["tmean"] = temp_average
        json_data["class"] = object_class
        json_data["hp_counter"] = hp_contour
        json_data["rp_counter"] = rp_contour
        file_data["facilities"].append(writeJson2(json_data))

        # file_data["facilities"].append(object_data)
        
    # with open('./json_rb/'+fname+'.json', 'w', encoding='utf-8') as make_file:
    with open(os.path.join(out_dir, (fname+'.json')), 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, indent="\t", ensure_ascii=False)

def analyze_dir_recursive():
    root_data_dir = '/home/inseo/Desktop/test_data'
    count = 1
    for data_dir in glob.glob(os.path.join(root_data_dir, 'FLIR*')):
        print(count)
        print(data_dir)
        
        # data_dir = '/home/inseo/Desktop/test_data/FLIR3572'
        out_dir = os.path.join(data_dir, 'json_rb')

        prefix_xml = 'Annotation'
        prefix_csv = 'csv_celsius'
        
        xml_dir = os.path.join(data_dir, prefix_xml)
        csv_dir = os.path.join(data_dir, prefix_csv)

        #create output path
        os.makedirs(out_dir, exist_ok=True)

        for item in glob.glob(os.path.join(xml_dir, '*.xml')):
            item_basename = os.path.basename(os.path.splitext(item)[0])
            csv = os.path.join(csv_dir, (item_basename+'.csv'))
            analyze(item, csv, out_dir)
            pass
        count += 1

if __name__=="__main__":
    #example
    xml = "/home/inseo/Desktop/test_data/FLIR2283/Annotation/frame00001.xml"
    csv = "/home/inseo/Desktop/test_data/FLIR2283/csv_celsius/frame00001.csv"
    out_dir = "/home/inseo/Desktop"
    analyze(xml, csv, out_dir)
    