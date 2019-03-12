try:
    import unzip_requirements # from the lambda layer "pytorch-v1-py36". Provides all PyTorch dependencies to the tmp folder to overcome the 250MB size limit.
except ImportError:
    pass

import os, io, json, tarfile, glob, time, logging, base64, boto3, PIL, torch
import torch.nn.functional as F
from torchvision import transforms

s3 = boto3.client('s3')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get('S3_BUCKET')
logger.info(f'S3 Bucket is {S3_BUCKET}')

MODEL = os.environ.get('MODEL')
logger.info(f'Model tar file is {MODEL}')

def load_model():
    """Loads the PyTorch model and the classes into memory from a tar.gz file on S3."""

    tmp_dir = '/tmp/pytorch-serverless'
    local_model = f'{tmp_dir}/model.tar.gz'
    os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f'Loading {MODEL} from S3 bucket {S3_BUCKET} to {local_model}')
    s3.download_file(S3_BUCKET, MODEL, local_model)
    tarfile.open(local_model).extractall(tmp_dir)
    os.remove(local_model)
    classes = open(f'{tmp_dir}/classes', 'r').read().splitlines()
    logger.info(f'Classes are {classes}')    
    model_path = glob.glob(f'{tmp_dir}/*_jit.pth')[0]
    logger.info(f'Model path is {model_path}')
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    return model.eval(), classes

model, classes = load_model()

def predict(model, classes, image_tensor):
    """Predicts the class of an image_tensor."""

    start_time = time.time()
    predict_values = model(image_tensor)
    logger.info("Inference time: {} seconds".format(time.time() - start_time))
    softmaxed = F.softmax(predict_values, dim=1)
    probability_tensor, index = torch.max(softmaxed, dim=1)
    prediction = classes[index]
    probability = "{:1.2f}".format(probability_tensor.item())
    logger.info(f'Predicted class is {prediction} with a probability of {probability}')
    return {'class': prediction, 'probability': probability}

preprocess_pipeline = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
def image_to_tensor(preprocess_pipeline, body):
    """Transforms the posted image to a PyTorch Tensor."""
    
    data = json.loads(body)
    name = data['name']
    image = data['file']
    dec = base64.b64decode(image)
    img = PIL.Image.open(io.BytesIO(dec))
    img_tensor = preprocess_pipeline(img)
    img_tensor = img_tensor.unsqueeze(0) # 3d to 4d for batch
    return img_tensor
    
def lambda_handler(event, context):
    """The main function which is called in the lambda function as defined in our template.yml"""

    image_tensor = image_to_tensor(preprocess_pipeline, event['body'])
    response = predict(model, classes, image_tensor)
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }
