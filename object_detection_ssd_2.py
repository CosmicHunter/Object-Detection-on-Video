import imageio
import cv2
import torch
from torch.autograd import Variable
from data import BaseTransform,VOC_CLASSES as labelmap
from ssd import build_ssd

def detect_obj(image,neural_net,transform_to_correct_format):
    height,width = image.shape[:2]
    image_transform = transform_to_correct_format(image)[0]
    X = torch.from_numpy(image_transform).permute(2,0,1)
    X = Variable(X.unsqueeze(0))
    Y = neural_net(X)
    detections = Y.data
    scaled_tensor = torch.Tensor([width,height,width,height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            c = (detections[0,i,j,1:] * scaled_tensor)
            c = c.numpy()
            cv2.rectangle(image,(int(c[0]),int(c[1])),(int(c[2]),int(c[3])),(0,255,0),2)
            cv2.putText(image,labelmap[i-1],(int(c[0]),int(c[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            j = j+1
    return image


nn_obj = build_ssd("test")
nn_obj.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

transform = BaseTransform(nn_obj.size, (104/256.0, 117/256.0, 123/256.0))


reader = imageio.get_reader('horse_vid.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output_vid_2.mp4',fps = fps)

for i,frame in enumerate(reader):
    new_frame = detect_obj(frame,nn_obj,transform)
    writer.append_data(new_frame)
    print(i)
    
writer.close()