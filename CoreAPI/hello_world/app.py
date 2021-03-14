import json
import boto3
import os
import pandas as pd
import math
import numpy as np

# ENDPOINT_NAME_LEFT = '<YOUR MODEL ENDPOINT NAME>'
ENDPOINT_NAME_RIGHT = '<YOUR MODEL ENDPOINT NAME>'

runtime= boto3.client('runtime.sagemaker')
lambda_client = boto3.client('lambda')

d_threshold = 0.1
null_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


connections = [
    (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)
]

labels = {
    "0" : "me", 
    "1" : "you", 
    "2" : "hello", 
    "3" : "from",
    "4" : "good",
    "5" : "how",
    "6" : "university",
    "7" : "welcome",
    "8" : "hope",
    "9" : "like",
    "10" : "new",
    "11" : "people",
    "12" : "technology",
    "13" : "use",
    "14" : "voice",
    "15" : "create"
}

def generatePointVectors(points, previousFrames):
    vectors = []

    prev_origin_x = 0
    prev_origin_y = 0

    dx = 0
    dy = 0

    if(len(previousFrames) == 0):
        prev_origin_x = 0
        prev_origin_y = 0
    else:
        prev_origin_x = previousFrames[0]
        prev_origin_y = previousFrames[1]
    
    origin_x, origin_y = points[0]

    origin_x_rounded = round((origin_x), 5)
    origin_y_rounded = round((origin_y), 5)

    dx = origin_x_rounded - prev_origin_x
    dy = origin_y_rounded - prev_origin_y

    vectors.append(origin_x_rounded)
    vectors.append(origin_y_rounded)

    for num, connection in enumerate(connections):

        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x_final = x1 - x0
        y_final = y1 - y0
        mag = math.sqrt((x_final)**2+(y_final)**2)

        x_vector = round((x_final/mag) + dx,5)
        y_vector = round((y_final/mag) + dy,5)

        vectors.append(x_vector)
        vectors.append(y_vector)

    return vectors

def generateCheckPoints(points):
    checkPoints = []

    palm_x, palm_y = points[0]
    thumb_x, thumb_y = points[4]
    index_x, index_y = points[8]
    pinky_x, pinky_y = points[20]

    mean_x = round((palm_x + thumb_x + index_x + pinky_x)/4, 5)
    mean_y = round((palm_y + thumb_y + index_y + pinky_y)/4, 5)

    checkPoints.append(mean_x)
    checkPoints.append(mean_y)

    return checkPoints

def checkPreviousFrame(currCheckPoints, prevCheckPoints):
    current_dx = currCheckPoints[0]
    current_dy = currCheckPoints[1]

    prev_dx = prevCheckPoints[0]
    prev_dy = prevCheckPoints[1]

    dx = round(abs(current_dx - prev_dx), 5)
    dy = round(abs(current_dy - prev_dy), 5)

    if(dx >= d_threshold or dy >= d_threshold):
        return True
    else:
        return False

def recalculateFrames(frames):
    
    cycledFrames = []
    cycledFrames.extend(frames)
    # Current Origin
    if(len(frames) > 12):
        base_x = cycledFrames[0]
        base_y = cycledFrames[1]

        secondFrame_dx = cycledFrames[12] - base_x
        secondFrame_dy = cycledFrames[13] - base_y

        # New Origin
        new_base_x = cycledFrames[12]
        new_base_y = cycledFrames[13]

        if(len(frames) > 24):
            thirdFrame_dx = cycledFrames[24] - base_x
            thirdFrame_dy = cycledFrames[25] - base_y

            # New second frame
            new_secondFrame_dx = cycledFrames[24] - new_base_x
            new_secondFrame_dy = cycledFrames[25] - new_base_y

            if(len(frames) > 36):
                fourthFrame_dx = cycledFrames[36] - base_x
                fourthFrame_dy = cycledFrames[37] - base_y

                # New third frame
                new_thirdFrame_dx = cycledFrames[36] - new_base_x
                new_thirdFrame_dy = cycledFrames[37] - new_base_y     
    
        i = 12
        while(i < 48):

            # This
            if(i >= 14 and i < 24 and len(frames) > 12):
                cycledFrames[i] = round((cycledFrames[i] - secondFrame_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - secondFrame_dy), 5)
            # This
            elif(i >= 26 and i < 36 and len(frames) > 24):
                original_keyframe_dx = cycledFrames[i] - thirdFrame_dx
                original_keyframe_dy = cycledFrames[i + 1] - thirdFrame_dy
                
                cycledFrames[i] = round(original_keyframe_dx + new_secondFrame_dx, 5)
                cycledFrames[i + 1] = round(original_keyframe_dy + new_secondFrame_dy, 5)
            # This
            elif(i >= 38 and i < 48 and len(frames) > 36):
                original_keyframe_dx = cycledFrames[i] - fourthFrame_dx
                original_keyframe_dy = cycledFrames[i + 1] - fourthFrame_dy
                
                cycledFrames[i] = round(original_keyframe_dx + new_thirdFrame_dx, 5)
                cycledFrames[i + 1] = round(original_keyframe_dy + new_thirdFrame_dy, 5)
            i = i + 2
    # 0 - 11
    # 12 - 23
    # 24 - 35
    # 36 - 47
    # Cycle out
    cycledFrames = cycledFrames[12:]
    return cycledFrames

def preprocessData(frames):
    
    dataToProcess = []
    dataToProcess.extend(frames)
    if(len(dataToProcess) != 48):
        if(len(dataToProcess) == 12):
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
        elif(len(dataToProcess) == 24):
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
        elif(len(dataToProcess) == 36):
            dataToProcess.extend(null_12)

    group_0 = []
    group_0.extend(dataToProcess[:12])
    group_0.extend(null_12)
    group_0.extend(null_12)
    group_0.extend(null_12)

    group_1 = []
    group_1.extend(dataToProcess[:24])
    group_1.extend(null_12)
    group_1.extend(null_12)

    group_2 = []
    group_2.extend(dataToProcess[:36])
    group_2.extend(null_12)

    group_3 = []
    group_3.extend(dataToProcess[:48])

    arr_0 = np.array(group_0)
    arr_1 = np.array(group_1)
    arr_2 = np.array(group_2)
    arr_3 = np.array(group_3)

    return arr_0, arr_1, arr_2, arr_3

def predict(set0, set1, set2, set3, ENDPOINT_NAME):

    df0 = pd.DataFrame(set0)
    df_T0 = df0.T
    csv_data0 = df_T0.to_csv(None, header=False, index=False)
    response0 = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=csv_data0)
    prob_list_0 = json.loads(response0['Body'].read().decode())
    
    df1 = pd.DataFrame(set1)
    df_T1 = df1.T
    csv_data1 = df_T1.to_csv(None, header=False, index=False)
    response1 = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=csv_data1)
    prob_list_1 = json.loads(response1['Body'].read().decode())
    
    df2 = pd.DataFrame(set2)
    df_T2 = df2.T
    csv_data2 = df_T2.to_csv(None, header=False, index=False)
    response2 = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=csv_data2)
    prob_list_2 = json.loads(response2['Body'].read().decode())
    
    df3 = pd.DataFrame(set3)
    df_T3 = df3.T
    csv_data3 = df_T3.to_csv(None, header=False, index=False)
    response3 = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=csv_data3)
    prob_list_3 = json.loads(response3['Body'].read().decode())
    
    max_prob_0 = np.amax(prob_list_0)
    max_prob_1 = np.amax(prob_list_1)
    max_prob_2 = np.amax(prob_list_2)
    max_prob_3 = np.amax(prob_list_3)

    out_label_0 = labels["{}".format(np.argmax(prob_list_0, axis=0))]
    out_label_1 = labels["{}".format(np.argmax(prob_list_1, axis=0))]
    out_label_2 = labels["{}".format(np.argmax(prob_list_2, axis=0))]
    out_label_3 = labels["{}".format(np.argmax(prob_list_3, axis=0))]

    label = out_label_0
    prob = max_prob_0
    
    if(prob < max_prob_1 and max_prob_1 > max_prob_2 and max_prob_1 > max_prob_3):
        prob = max_prob_1
        label = out_label_1
    elif(prob < max_prob_2 and max_prob_2 > max_prob_3 and max_prob_2 > max_prob_1):
        prob = max_prob_2
        label = out_label_2
    elif(prob < max_prob_3 and max_prob_3 > max_prob_1 and max_prob_3 > max_prob_2):
        prob = max_prob_3
        label = out_label_3

    return label, prob

def cleanUp(frames, model):
    temp_frames = []
    temp_frames.extend(frames)

    temp_label = ''
    temp_prob = 0

    if(model is None):
        temp_frames = []
        return temp_frames, temp_label, temp_prob

    while(len(temp_frames) != 0):

        temp_frames = recalculateFrames(temp_frames)
        if(len(temp_frames) != 0):
            # Preprocess
            set0, set1, set2, set3 = preprocessData(temp_frames)
            # Classify
            temp_label, temp_prob = predict(set0, set1, set2, set3, model)
    temp_frames = []
    return temp_frames, temp_label, temp_prob

def lambda_handler(event, context):
    
    # Call Mediapipe Function
    # Result is rightHandPoints and leftHandPoints
    results = json.loads(event)
    results = results['data']    
    
    imgBase64 = results['imgBase64']
    lambda_payload = {'body': imgBase64}
    lambda_payload = json.dumps(lambda_payload)
    mediaResponse = lambda_client.invoke(FunctionName='<Your Mediapipe API Lambda Function Name>', InvocationType='RequestResponse', Payload = lambda_payload)
    
    mediaResponse = json.loads(mediaResponse["Payload"].read())
    


    rightHandPoints = mediaResponse['body']['rightHandPoints']
    leftHandPoints = mediaResponse['body']['leftHandPoints']

    rightLabel = ''
    rightProb = 0
    leftLabel = ''
    leftProb = 0  
    
    rightKeyFrames = results['rightKeyFrames']
    leftKeyFrames = results['leftKeyFrames']

    rightKeyCheckPoints = results['rightKeyCheckPoints']
    leftKeyCheckPoints = results['leftKeyCheckPoints']

    rightVectors = []
    leftVectors = []

    rightCheckPoints = []
    leftCheckPoints = []

    # Both hands are detected
    if(len(rightHandPoints) != 0 and len(leftHandPoints) != 0):
        rightVectors = generatePointVectors(rightHandPoints, rightKeyFrames)
        rightCheckPoints = generateCheckPoints(rightHandPoints)
        
        leftVectors = generatePointVectors(leftHandPoints, leftKeyFrames)
        leftCheckPoints = generateCheckPoints(leftHandPoints)

        # For first frame
        if(len(rightKeyFrames) == 0 and len(leftKeyFrames) == 0):
            rightKeyFrames.extend(rightVectors)
            rightKeyCheckPoints.extend(rightCheckPoints)

            leftKeyFrames.extend(leftVectors)
            leftKeyCheckPoints.extend(leftCheckPoints)

            # Preprocess
            r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
            # Classify
            rightLabel, rightProb = predict(r_set0, r_set1, r_set2, r_set3, ENDPOINT_NAME_RIGHT)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }

        if(checkPreviousFrame(rightCheckPoints, rightKeyCheckPoints) or checkPreviousFrame(leftCheckPoints, leftKeyCheckPoints)):
            rightKeyCheckPoints = []
            leftKeyCheckPoints = []

            if(len(rightKeyFrames) == 48):
                rightKeyFrames = recalculateFrames(rightKeyFrames)
            if(len(leftKeyFrames) == 48):
                leftKeyFrames = recalculateFrames(leftKeyFrames)

            rightKeyFrames.extend(rightVectors)
            rightKeyCheckPoints.extend(rightCheckPoints)

            leftKeyFrames.extend(leftVectors)
            leftKeyCheckPoints.extend(leftCheckPoints)
            
            # Preprocess
            r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
            # Classify
            rightLabel, rightProb = predict(r_set0, r_set1, r_set2, r_set3, ENDPOINT_NAME_RIGHT)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }
        else:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }

    # Right hand detected only
    elif(len(rightHandPoints) != 0 and len(leftHandPoints) == 0):
        rightVectors = generatePointVectors(rightHandPoints, rightKeyFrames)
        rightCheckPoints = generateCheckPoints(rightHandPoints)

        # Cleanup Left hand
        if(len(leftKeyFrames) != 0):                                
            leftKeyFrames, leftLabel, leftProb = cleanUp(leftKeyFrames, None)

        # For first frame
        if(len(rightKeyFrames) == 0):
            rightKeyFrames.extend(rightVectors)
            rightKeyCheckPoints.extend(rightCheckPoints)

            # Preprocess
            r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
            # Classify
            rightLabel, rightProb = predict(r_set0, r_set1, r_set2, r_set3, ENDPOINT_NAME_RIGHT)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }

        if(checkPreviousFrame(rightCheckPoints, rightKeyCheckPoints)):
            rightKeyCheckPoints = []

            if(len(rightKeyFrames) == 48):
                rightKeyFrames = recalculateFrames(rightKeyFrames)

            rightKeyFrames.extend(rightVectors)
            rightKeyCheckPoints.extend(rightCheckPoints)

            # Preprocess
            r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
            # Classify
            rightLabel, rightProb = predict(r_set0, r_set1, r_set2, r_set3, ENDPOINT_NAME_RIGHT)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }
        else:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }

    # Left hand detected only
    elif(len(leftHandPoints) != 0 and len(rightHandPoints) == 0):
        leftVectors = generatePointVectors(leftHandPoints, leftKeyFrames)
        leftCheckPoints = generateCheckPoints(leftHandPoints)

        # Clean up right hand
        if(len(rightKeyFrames) != 0):                                
            rightKeyFrames, rightLabel, rightProb = cleanUp(rightKeyFrames, None)

        # For first frame
        if(len(leftKeyFrames) == 0):
            leftKeyFrames.extend(leftVectors)
            leftKeyCheckPoints.extend(leftCheckPoints)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }
        
        if(checkPreviousFrame(leftCheckPoints, leftKeyCheckPoints)):
            leftKeyCheckPoints = []

            if(len(leftKeyFrames) == 48):
                leftKeyFrames = recalculateFrames(leftKeyFrames)

            leftKeyFrames.extend(leftVectors)
            leftKeyCheckPoints.extend(leftCheckPoints)
        
        return {
                "statusCode": 200,
                "body": json.dumps({
                    "rightKeyFrames" : rightKeyFrames,
                    "leftKeyFrames" : leftKeyFrames,
                    "rightKeyCheckPoints": rightKeyCheckPoints,
                    "leftKeyCheckPoints": leftKeyCheckPoints,
                    "label" : rightLabel,
                    "prob" : rightProb
                }),
            }