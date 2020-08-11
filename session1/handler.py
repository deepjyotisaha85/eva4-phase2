try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import tarfile
import io
import base64
import json
from requests_toolbelt.multipart import decoder


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'deep-eva4-a1-mobilenet'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenetv2.pt'

s3 = boto3.client('s3')

try:
    print("In try block...")
    #if os.path.isfile(MODEL_PATH) != True:
    print("In try block - True...")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    bytestream = io.BytesIO(obj['Body'].read())
    print("loading model...")
    model = torch.jit.load(bytestream)
    print("loaded model...")
except Exception as e:
    print("Could not find model...")
    print(repr(e))
    raise(e)    


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    print("transforming image...")
    tensor = transform_image(image_bytes=image_bytes)
    print("transformed image.. calling model..")
    return model(tensor).argmax().item()


def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print("getting prediction...")
        prediction = get_prediction(image_bytes=picture.content)
        print("got prediction...")

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }