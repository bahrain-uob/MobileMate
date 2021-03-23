# MobileMate

This project provides an API that enables the features of PoseMate to be used in mobile and web applications. The API is split in two parts:
- Core 
- Mediapipe

The Core API is the main access point which invokes the Mediapipe API. The Mediapipe API consists of the Mediapipe SDK which provides the hand tracking. The output of this hand tracking is sent back to the Core API for further processing and then classifying American Sign Language gestures.

*If you'd like to know more about how our sign language translation works and how to train your own model, check out our [PoseMate](https://github.com/bahrain-uob/PoseMate) repository.*

To get started, deploy each of these API's starting with the Mediapipe API first. You will find instructions to build and deploy these API's within each of their directories.

Before deploying the CoreAPI, update the following in its [app.py](./CoreAPI/hello_world/app.py):

>ENDPOINT_NAME_RIGHT</br>
>ENDPOINT_NAME_LEFT</br>
>MediapipeAPI Lambda Function Name

## Giving Permssions to the CoreAPI

In the **IAM** Console, create a new policy and give it a suitable name, for example: `MediapipeAPIPolicy`. Edit the policy and add the following to it:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowAuroraToExampleFunction",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": "<YOUR MEDIAPIPE API LAMBDA FUNCTION ARN>"
        }
    ]
}
```

Replace the resource with the ARN of your Mediapipe API Lambda Function.

Head over to the **AWS Lambda** Console, find your CoreAPI lambda function. Under the **Permissions** tab, click on the role.
In the **IAM** Console, add the following policies to it:
-  **AWSLambdaBasicExecutionRole** policy
-  **AmazonSagemakerFullAccess** policy
-  **MediapipeAPIPolicy** policy

## Integrations

These API's can be integrated into a mobile app simply by invoking the CoreAPI Lambda function.