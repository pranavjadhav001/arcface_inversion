# arcface_inversion
Code for generating class representations using pretrained ArcFace Model for explainablity 
https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/random.png?raw=true
![alt text](https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/arcface_inversion_diagram.png?raw=true)

## Steps to execute :
- pull latest pytorch docker image from docker hub
- docker run -p 8080:8080 --rm -v arcface_inversion/:/arcface_inversion/ -it --gpus=all pytorch/pytorch:latest bash
- pip3 install -r <b>requirements.txt</b>
- apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
- jupyter notebook --port=8080 --ip=0.0.0.0 --allow-root --no-browser

## Steps for performing:
- Train embedding model using <b>ArcFace loss</b>
- Here I'm using <b>pytorch-metric-learning </b> library(https://kevinmusgrave.github.io/pytorch-metric-learning/) for using high level api
- I'm using <b>Resnet-18 SE block model</b> architecture(https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/resnet.py)
- Train the deep metric learning task on any dataset, using MNIST here in this example
- Save the embedding model weights , along with ArcFace loss weights
- Get pretrained Batch Normalization priors, store them as variables
- Initialize random gaussian centered images once
- Run training loop for 20k iterations, and update the input image every iteration using statistic loss ,Arcface loss and regularization loss<br/>
![alt text](https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/algo.png?raw=true)
- Statistic loss is calculated using <b>running mean and variance</b> of all batch norm layers in the model
![alt text](https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/formula.png?raw=true)
- You can tune <b>weight decay , alpha , learning rate, epochs, batch size</b> for different/better results.

## Results

### Target Label 0
![alt text](https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/0.png?raw=true)
### Target Label 1
![alt text](https://github.com/pranavjadhav001/diffusers_hairstyle_transfer/blob/main/images/1.png?raw=true)

## Notes
- Complex dataset can show better generation of class representations
- Pretrained Embedding model is not enough to reproduce, you will need arcface class weights as well
- Resnet-50 is used in paper but here a flavour of resnet-18 is used
- Image Background matters : <br/> 
<em>"data-free method employing the BN priors to restore ImageNet
images for distillation, quantization and pruning. Their model
inversion results contain obvious artifact in the background due
to the translation augmentation during training. By contrast, our
ArcFace model is trained on normalized face crops without back-
ground, thus the restored faces exhibit less artifact."</em>

## References
- https://github.com/ronghuaiyang/arcface-pytorch
- MNIST training example using PML : https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples
- Arcface paper : https://arxiv.org/abs/1801.07698
