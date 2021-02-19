import json
import boto3
import os
import pickle
import math
import pandas as pd
import numpy as np

ENDPOINT_NAME = 'xgboost-endpoint'
runtime= boto3.client('runtime.sagemaker')


nullVectors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
null_24 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

d_threshold = 0.1

connections = [
    (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)
]


labels = {
            "0" : "A", "1" : "B", "2" : "C", "3" : "D", "4" : "E",
            "5" : "F", "6" : "G", "7" : "H", "8" : "I", "9" : "K",
            "10" : "L", "11" : "M", "12" : "N", "13" : "O", "14" : "P",
            "15" : "Q", "16" : "R", "17" : "S", "18" : "T", "19" : "U",
            "20" : "V", "21" : "W", "22" : "X", "23" : "Y"
        }

def generatePointVectors(rightPoints, leftPoints, previousFrames):
    rightVectors = []
    leftVectors = []

    r_prev_origin_x = 0
    r_prev_origin_y = 0
    l_prev_origin_x = 0
    l_prev_origin_y = 0

    r_dx = 0
    r_dy = 0
    l_dx = 0
    l_dy = 0

    if(len(previousFrames) == 0):
        r_prev_origin_x = 0
        r_prev_origin_y = 0
        l_prev_origin_x = 0
        l_prev_origin_y = 0
    else:
        r_prev_origin_x = previousFrames[0]
        r_prev_origin_y = previousFrames[1]
        l_prev_origin_x = previousFrames[12]
        l_prev_origin_y = previousFrames[13]

    if(len(rightPoints) != 0):
        r_origin_x, r_origin_y = rightPoints[0]

        r_origin_x_rounded = round((r_origin_x), 5)
        r_origin_y_rounded = round((r_origin_y), 5)

        r_dx = r_origin_x_rounded - r_prev_origin_x
        r_dy = r_origin_y_rounded - r_prev_origin_y

        rightVectors.append(r_origin_x_rounded)
        rightVectors.append(r_origin_y_rounded)

    if(len(leftPoints) != 0):
        l_origin_x, l_origin_y = leftPoints[0]

        l_origin_x_rounded = round((l_origin_x), 5)
        l_origin_y_rounded = round((l_origin_y), 5)

        l_dx = l_origin_x_rounded - l_prev_origin_x
        l_dy = l_origin_y_rounded - l_prev_origin_y

        leftVectors.append(l_origin_x_rounded)
        leftVectors.append(l_origin_y_rounded)

    for num, connection in enumerate(connections):

        if(len(rightPoints) != 0):
            r_x0, r_y0 = rightPoints[connection[0]]
            r_x1, r_y1 = rightPoints[connection[1]]
            r_x_final = r_x1 - r_x0
            r_y_final = r_y1 - r_y0
            r_mag = math.sqrt((r_x_final)**2+(r_y_final)**2)

            r_x_vector = round((r_x_final/r_mag) + r_dx,5)
            r_y_vector = round((r_y_final/r_mag) + r_dy,5)

            rightVectors.append(r_x_vector)
            rightVectors.append(r_y_vector)

        if(len(leftPoints) != 0):
            l_x0, l_y0 = leftPoints[connection[0]]
            l_x1, l_y1 = leftPoints[connection[1]]
            l_x_final = l_x1 - l_x0
            l_y_final = l_y1 - l_y0
            l_mag = math.sqrt((l_x_final)**2+(l_y_final)**2)

            l_x_vector = round((l_x_final/l_mag) + l_dx,5)
            l_y_vector = round((l_y_final/l_mag) + l_dy,5)

            leftVectors.append(l_x_vector)
            leftVectors.append(l_y_vector)
            
    finalVectors = []
    if(len(rightVectors) != 0 and len(leftVectors) != 0):
        finalVectors.extend(rightVectors)
        finalVectors.extend(leftVectors)
    if(len(rightVectors) == 0):
        finalVectors.extend(nullVectors)
        finalVectors.extend(leftVectors)
    if(len(leftVectors) == 0):
        finalVectors.extend(rightVectors)
        finalVectors.extend(nullVectors)

    return finalVectors

def generateCheckPoints(rightPoints, leftPoints):
    checkPoints = []
    if(len(rightPoints) != 0 and len(leftPoints) != 0):
        r_palm_x, r_palm_y = rightPoints[0]
        r_thumb_x, r_thumb_y = rightPoints[4]
        r_index_x, r_index_y = rightPoints[8]
        r_pinky_x, r_pinky_y = rightPoints[20]

        r_mean_x = round((r_palm_x + r_thumb_x + r_index_x + r_pinky_x)/4, 5)
        r_mean_y = round((r_palm_y + r_thumb_y + r_index_y + r_pinky_y)/4, 5)

        l_palm_x, l_palm_y = leftPoints[0]
        l_thumb_x, l_thumb_y = leftPoints[4]
        l_index_x, l_index_y = leftPoints[8]
        l_pinky_x, l_pinky_y = leftPoints[20]

        l_mean_x = round((l_palm_x + l_thumb_x + l_index_x + l_pinky_x)/4, 5)
        l_mean_y = round((l_palm_y + l_thumb_y + l_index_y + l_pinky_y)/4, 5)

        checkPoints.append(r_mean_x)
        checkPoints.append(r_mean_y)
        checkPoints.append(l_mean_x)
        checkPoints.append(l_mean_y)

    elif(len(rightPoints) != 0 and len(leftPoints) == 0):
        r_palm_x, r_palm_y = rightPoints[0]
        r_thumb_x, r_thumb_y = rightPoints[4]
        r_index_x, r_index_y = rightPoints[8]
        r_pinky_x, r_pinky_y = rightPoints[20]

        r_mean_x = round((r_palm_x + r_thumb_x + r_index_x + r_pinky_x)/4, 5)
        r_mean_y = round((r_palm_y + r_thumb_y + r_index_y + r_pinky_y)/4, 5)

        checkPoints.append(r_mean_x)
        checkPoints.append(r_mean_y)
        checkPoints.append(0)
        checkPoints.append(0)
    elif(len(leftPoints) != 0 and len(rightPoints) == 0):
        l_palm_x, l_palm_y = leftPoints[0]
        l_thumb_x, l_thumb_y = leftPoints[4]
        l_index_x, l_index_y = leftPoints[8]
        l_pinky_x, l_pinky_y = leftPoints[20]

        l_mean_x = round((l_palm_x + l_thumb_x + l_index_x + l_pinky_x)/4, 5)
        l_mean_y = round((l_palm_y + l_thumb_y + l_index_y + l_pinky_y)/4, 5)

        checkPoints.append(0)
        checkPoints.append(0)
        checkPoints.append(l_mean_x)
        checkPoints.append(l_mean_y)

    return checkPoints

def checkPreviousFrame(currCheckPoints, prevCheckPoints):

    r_current_dx = currCheckPoints[0]
    r_current_dy = currCheckPoints[1]
    l_current_dx = currCheckPoints[2]
    l_current_dy = currCheckPoints[3]

    r_prev_dx = prevCheckPoints[0]
    r_prev_dy = prevCheckPoints[1]
    l_prev_dx = prevCheckPoints[2]
    l_prev_dy = prevCheckPoints[3]

    r_dx = round(abs(r_current_dx - r_prev_dx), 5)
    r_dy = round(abs(r_current_dy - r_prev_dy), 5)
    l_dx = round(abs(l_current_dx - l_prev_dx), 5)
    l_dy = round(abs(l_current_dy - l_prev_dy), 5)

    if(r_dx >= d_threshold or r_dy >= d_threshold or l_dx >= d_threshold or l_dy >= d_threshold):
        return True
    else:
        return False

def recalculateFrames(frames):
    
    cycledFrames = []
    cycledFrames.extend(frames)
    # Current Origin

    #right hand origins
    r_base_x = cycledFrames[0]
    r_base_y = cycledFrames[1]

    r_24_dx = cycledFrames[24] - r_base_x
    r_25_dy = cycledFrames[25] - r_base_y

    r_48_dx = cycledFrames[48] - r_base_x
    r_49_dy = cycledFrames[49] - r_base_y

    r_72_dx = cycledFrames[72] - r_base_x
    r_73_dy = cycledFrames[73] - r_base_y

    #left hand origins
    l_base_x = cycledFrames[12]
    l_base_y = cycledFrames[13]

    l_36_dx = cycledFrames[36] - l_base_x
    l_37_dy = cycledFrames[37] - l_base_y

    l_60_dx = cycledFrames[60] - l_base_x
    l_61_dy = cycledFrames[61] - l_base_y

    l_84_dx = cycledFrames[84] - l_base_x
    l_85_dy = cycledFrames[85] - l_base_y

    # New Origin
    new_r_base_x = cycledFrames[24]
    new_r_base_y = cycledFrames[25]

    new_r_48_x = cycledFrames[48] - new_r_base_x
    new_r_49_y = cycledFrames[49] - new_r_base_y
    
    new_r_72_x = cycledFrames[72] - new_r_base_x
    new_r_73_y = cycledFrames[73] - new_r_base_y

    new_l_base_x = cycledFrames[36]
    new_l_base_y = cycledFrames[37]

    new_l_60_x = cycledFrames[60] - new_l_base_x
    new_l_61_y = cycledFrames[61] - new_l_base_y
    
    new_l_84_x = cycledFrames[84] - new_l_base_x
    new_l_85_y = cycledFrames[85] - new_l_base_y

    i = 24
    while(i < 96):
        if(i >= 26 and i < 36):
            cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
            cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
        elif(i >= 38 and i < 48):
            cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
            cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
            
        elif(i >= 50 and i < 60):
            r_orignial_keyframe_x = cycledFrames[i] - r_48_dx
            r_orignial_keyframe_y = cycledFrames[i + 1] - r_49_dy
            
            cycledFrames[i] = round(r_orignial_keyframe_x + new_r_48_x, 5)
            cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_49_y, 5)
        elif(i >= 62 and i < 72):
            l_orignial_keyframe_x = cycledFrames[i] - l_60_dx
            l_orignial_keyframe_y = cycledFrames[i + 1] - l_61_dy
            
            cycledFrames[i] = round(l_orignial_keyframe_x + new_l_60_x, 5)
            cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_61_y, 5)
        elif(i >= 74 and i < 84):
            r_orignial_keyframe_x = cycledFrames[i] - r_72_dx
            r_orignial_keyframe_y = cycledFrames[i + 1] - r_73_dy
            
            cycledFrames[i] = round(r_orignial_keyframe_x + new_r_72_x, 5)
            cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_73_y, 5)
        elif(i >= 86 and i < 96):
            l_orignial_keyframe_x = cycledFrames[i] - l_84_dx
            l_orignial_keyframe_y = cycledFrames[i + 1] - l_85_dy
            
            cycledFrames[i] = round(l_orignial_keyframe_x + new_l_84_x, 5)
            cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_85_y, 5)
        i = i + 2
    # 0 - 23
    # 24 - 47
    # 48 - 71
    # 72 - 95
    # Cycle out
    cycledFrames = cycledFrames[24:]
    return cycledFrames

def predict(frames):
    dataToProcess = []
    dataToProcess.extend(frames)
    if(len(dataToProcess) != 96):
        if(len(dataToProcess) == 24):
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
        elif(len(dataToProcess) == 48):
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
        elif(len(dataToProcess) == 72):
            dataToProcess.extend(null_24)
        else:
            print("Error in preprocessData. Length of dataToProcess: ", len(dataToProcess))
    # Convert values to DMatrix format
    
    df = pd.DataFrame(dataToProcess)
    df_T = df.T
    csv_data = df_T.to_csv(None, header=False, index=False)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=csv_data)
    result = json.loads(response['Body'].read().decode())
    max_prob = np.amax(result)
    pred_label_idx = np.argmax(result, axis=0)
    out_label = labels["{}".format(pred_label_idx)]
    return out_label


def lambda_handler(event, context):
    # Parse input
    label = ""
    results = json.loads(event)
    results = results['data']
    rightHandPoints = results['right']
    leftHandPoints = results['left']
    keyFrames = results['keyFrames']
    keyCheckPoints = results['keyCheckPoints']
    
    
    finalVectors = generatePointVectors(rightHandPoints, leftHandPoints, keyFrames)
    
    checkPoints = generateCheckPoints(rightHandPoints, leftHandPoints)
    if(len(keyFrames)==0):
        keyFrames.extend(finalVectors)
        keyCheckPoints.extend(checkPoints)
        # label = predict(keyFrames
        return {
            "statusCode": 200,
            "body": json.dumps({
                "keyFrames" : keyFrames ,
                "keyCheckPoints": keyCheckPoints,
                "label" : label
            }),
        }
    if (checkPreviousFrame(checkPoints, keyCheckPoints)):
        keyCheckPoints = []
        if(len(keyFrames)==96):
            keyFrames = recalculateFrames(keyFrames)
        keyFrames.extend(finalVectors)
        keyCheckPoints.extend(checkPoints)
        # label = predict(keyFrames)
        return {
            "statusCode": 200,
            "body": json.dumps({
                "keyFrames" : keyFrames ,
                "keyCheckPoints": keyCheckPoints,
                "label" : label
            }),
        }
        
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "keyFrames" : keyFrames ,
                "keyCheckPoints": keyCheckPoints,
                "label" : label
            }),
        }