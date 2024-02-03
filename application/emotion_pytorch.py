from scipy.special import softmax
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.nn.functional as F




from torchvision import datasets, transforms
# from tqdm.notebook import tqdm

import timm    #←これを追加
import cv2
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detection_model = cv2.dnn.readNetFromCaffe('.\\models\\deploy.prototxt.txt', '.\\models\\res10_300x300_ssd_iter_140000_fp16.caffemodel')
model_names = timm.list_models(pretrained=True)
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=7)
model.to(device)
labels =['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model.load_state_dict(torch.load('.\\model\\model_5e_4.pth', map_location=torch.device(device)))
model.to('cpu')
def getColor(label):
    if label == "angry":
        color = (0,0,255)

    elif label == 'disgust':
        color = (0,190,246)
    elif label == 'fear':
        color = (72,50,72)
    elif label=='happy':
        color = (203,192,255)
    elif label=='neutral':
        color = (0,255,0)
    elif label=='sad':
        color = (255,0,0)
    elif label=='surprise':
        color = (255,255,255)

    return color

def face_mask_prediction(img):
    # step 1: face detection
    # img = cv2.imread('./images/IMG_20240131_172337.png')
    # img= cv2.resize(img,(480,640))
    # print(img.shape)
    image = img.copy()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1,(300,300),swapRB=True)

    face_detection_model.setInput(blob)
    detection = face_detection_model.forward() # it will give us the detection
    for i in range(0,detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.5:
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            box = box.astype(int)
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            #cv2.rectangle(image,pt1,pt2,(0,255,0),1) #the rectangle box around the face

            #step 2: data prepeocessing
            #need to crop the face

            face = image[box[1]:box[3],box[0]:box[2]]
            face_blob = cv2.dnn.blobFromImage(face,1,(224,224),swapRB=True)
            face_blob_squeeze = np.squeeze(face_blob).T #for correct rotation .T
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE) #for correct structure
            face_blob_flip = cv2.flip(face_blob_rotate,1) #for flip the image

            # normalization
            img_norm = np.maximum(face_blob_flip,0)
            # img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()
            transformer= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean,std)
            ])

            # plt.imshow(img_norm)
            img_input = transformer(np.uint8(img_norm))
            img_input=img_input.view(1,3,224,224)
            # print(img_input.shape)
            # img_input=transforms.ToTensor()(img_input)
            result = model(img_input)

            #print(result) #the probabilities of the labels

            with torch.no_grad():
                result = softmax(result.to('cpu'))[0]
            # print(result)
            confidence_index = result.argmax() #take out the labels out of this and exctract only where we have the highest value and that means that it wears a mask

            confidence_score = result[confidence_index]
            # print('The confidence score is =',confidence_score)
            label = labels[confidence_index] #label out
            label_text = '{}: {:,.0f} %'.format(label,confidence_score*100) #these will print only the integer values
            # print(label_text) #shows if it wears a mask or no and the probability of this
            res_values=result.to('cpu').detach().numpy().copy()*100
            # print(res_values)

            # put the ractangular box and whow the label on top of the face

            color = getColor(label)
            cv2.rectangle(image,pt1,pt2,color,1)
            cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,2,color,2) #thickness is 2 to be more clear the text on top of the face
            # plt.imshow(image)
            for i in range(7):
                cv2.rectangle(image, (80, 300+10*i), (80+int(res_values[i]), 310+10*i), getColor(labels[i]), thickness=-1)
                cv2.putText(image,labels[i],(10, 307+10*i), cv2.FONT_HERSHEY_PLAIN,0.9, getColor(labels[i]),1)
            # interval=13
            # for i in range(7):
            #     cv2.rectangle(image, (120, 300+interval*i), (100+int(res_values[i]), 310+interval*i), getColor(labels[i]), thickness=-1)
            #     cv2.putText(image,labels[i],(10, 310+interval*i), cv2.FONT_HERSHEY_PLAIN,1.5, getColor(labels[i]),2)
            return image,result.to('cpu').detach().numpy().copy()





