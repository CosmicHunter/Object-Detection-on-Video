
"""
# =============================================================================
#  The folder contains the data folder , this folder contains the
   Classes BaseTransform that will do the required transformations
   so that the input images will be compatible with the neural net.
   
   Layers is a folder that contains tools for detection and multibox
   parts of the ssd
   
   
   We have the ssd.py file in the working folder that contains the 
   architecture of the single shot multibox detector
   this is taken from the https://github.com/amdegroot/ssd.pytorch 
   repository. Its the pytorch implementation of the ssd.
   
   There is one more file that is ssd300_mAP_.... that contained the 
   weights of the pretrained ssd model. We will simply load this file
   to do the object detection using the pretrained model.
   We will transfer these weights to the model we implement.
   
   ## Libraries
   
   torch.autograd contains Variable class that will be used to  convert
   tensors  into some torch variables that will contain both the tensor and
   the gradient and this torch variable containing both gradient
   and tensor will be like one element of the graph.
   
   Base Transform is a class that will do the required transformations
   so that the input images will be compatible with the neural network
   When we feed the images to the neural network they have to have a 
   certain format. So BaseTransform class from data takes care of that.
   
   VOC_CLASSES is a just a dictionary that will do the encoding of the classes
   Different classes will be encoded as numbers.
   
   
   build_ssd will be the constructor of the neural network
# =============================================================================
"""

import cv2
import torch
from torch.autograd import Variable
from data import BaseTransform,VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# We will do a frame by frame detection . The detect function that we will implement will do the detection on single images coming from a video

# This function will return the original images with detection rectangles

def detect_object(original_image,neural_net, transform_to_correct_format):
     # Getting the height and width of the image
     height,width= original_image.shape[:2]   # at Index 2 is the number of channels  , we only want height and width of the image
     transform_img = transform_to_correct_format(original_image)[0]  # It returns two elements and we are only interested in the first element.
     # Transform Image has the right dimensions and the right color value
     # transform_img is still a numpy array
     
     """ 2nd Transformation is to convert this numpy array into a torch tensor
     """
     X = torch.from_numpy(transform_img)
     """ The neural network ssd was trained on the images that were GRB  (Green Red Blue) so we need to convert our input from RGB to 
         GRB
     """
     X = X.permute(2,0,1)  # 0 is Red , 1 is Blue , 2 is Green .
     
     """
     Third Transformation is to add a fake dimension to our input corresponding to the batch . The reason for doing this is that the neural net
     cannot accept single input image. It only accepts them into some batches. So now we need to create a structure in which the first
     dimension correspond to the batch and the other to the input.
     
     We use the unsqeeze function to create the fake dimension
     """
     X = Variable(X.unsqueeze(0)) # Unsqeeze function takes the index of the dimension that we need to add and batch should always be first dimension.
     
     # Final transformation is to convert it into a torch variable
     # Now we are ready to pass this as input to the pretrained neural network ssd . The neural network is already pretrained to detect 30-40 objects   
    
     # Feeding the torch variable (that contains both the tensor and the gradient ) into the neural network
     Y = neural_net(X) 
     """
     Y is the output. We will create a tensor named detections and get its value from the tensor contained in the Y
     """
     detections  = Y.data
    # We need to create a new tensor that has the dimensions (width,height,width,height) that's because the positions of the detected objects
    # inside the image has to be normalised between 0 and 1 and to this we need this scaled tensor with four dimensions.
     scaled_tensor = torch.Tensor([width,height,width,height])
    # The first width height correspond to the upper left corner of the detected rectangle and the other two to the lower right corner of the rectangle
    
     """
    The detections tensor contains four elements namely [batch , number of classes(i.e the number of objects that can be detected) ,
                                                         number of occurances of the class , tuple of 5 elements (score,x0,y0,x1,y1)
                                                         for each occurance of each class in the batch we will get a score for this occurance
                                                         and a cordinate of the upper left corner detecting the occurance and the lower right corner
                                                         These scores are such that if the score is less than 0.6 than the occrance of the class in 
                                                         that image will not be found.]
     """
    
     """ 
   Now we need to make a for loop that will iterate through all the classes and then through all the occurance of these classes , we are going to
   look for certain number of occurances for each class , we will get the score and check if the score is higher and lower than the threshold. if 
   the score is higher than the specified threshold we keep the occurance else we reject it.
     
   """
     for object_class in range(detections.size(1)):
         occurrence_of_object_class = 0
         
         # score is detections[0,object_class,occurrence_of_object_class,0]
         while detections[0,object_class,occurrence_of_object_class,0] >= 0.6:
             # We will get the cordinates for its rectangle
             """
             The cordinates need to be scaled so we will multiply them with our scaled_tensor(which did normalisation btw 0 and 1) in order 
             to scale them w.r.t image dimensions
             Also to draw rectangles we need to convert them to numpy array as they are tensors . because open-cv expects arguments as a numpy array.
             """
             cordinates = (detections[0,object_class,occurrence_of_object_class,1:] * scaled_tensor) # cordinates at the scale of the image
             cordinates = cordinates.numpy()
             
             # Draw rectangle
             cv2.rectangle(original_image,(int(cordinates[0]),int(cordinates[1])),(int(cordinates[2]),int(cordinates[3])),(0,255,0),2)
             # Draw Label
             cv2.putText(original_image,labelmap[object_class-1]  ,(int(cordinates[0]),int(cordinates[1])),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
             occurrence_of_object_class+=1
     
     return original_image

"""
# =============================================================================
#   Python: cv.PutText(img, text, org, font, color) → None
    Parameters:	

        img – Image.
        text – Text string to be drawn from labelmap which is a dictionary . we will pick the class name string from it.
        org – Bottom-left corner of the text string in the image. We have chosen top left corner for displaying label
        fontFace – Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID’s can be combined with FONT_ITALIC to get the slanted letters.
        fontScale – Font scale factor that is multiplied by the font-specific base size.
        color – Text color.
        thickness – Thickness of the lines used to draw a text.
        lineType – Line type. See the line for details. (cv2.LINE_AA to display the text continuously and not dot-dot.)
       
# =============================================================================
            
"""

# Creating the SSD Neural Network
neural_net_object = build_ssd("test")  # Build ssd expects arguments like test,train and here we will not train the model, instead we will load 
# the weigths of the pretrained model. Which means we only need to test it on our video file.

"""
ssd300_mAP_77.43_v2.pth is a very powerful pretrained model that is able to detect many objects
"""

neural_net_object.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 
# load state dict adds the wts to our neural net object and torch.load will open a Tensor which will contain these wts.
# load_state_dict : Loads a model’s parameter   dictionary using a deserialized state_dict

""" state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
    When saving a model for inference, it is only necessary to save the trained model’s learned parameters. Saving the model’s state_dict
    with the torch.save() function will give you the most flexibility for restoring the model later, which is why it is the recommended method 
    for saving models.A common PyTorch convention is to save models using either a .pt or .pth file extension.
    
    Parameters for torch.load :-
    
    f – a file-like object (has to implement read(), :meth`readline`, :meth`tell`, and :meth`seek`), or a string containing a file name

    map_location – a function, torch.device, string or a dict specifying how to remap storage locations

    pickle_module – module used for unpickling metadata and objects (has to match the pickle_module used to serialize file)

"""

    
# Creating the transformation
# We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input 
# of the neural network.
# neural_net_object.size is the target size of the images that we will feed to the neural network 
# The triplet is the scale under which neural net was trained. These are some scale values to ensure the color values are in the right scale
transform = BaseTransform(neural_net_object.size, (104/256.0, 117/256.0, 123/256.0))    

# Now we will perform the object detection on the video.

reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']

# Creating an output video with object detection with the above fps
writer = imageio.get_writer('output_vid.mp4',fps = fps)

# Now we will process each of the frames of the video and apply our detect_object function

for i,frame in enumerate(reader):
    # Each time we take the reader frame and apply detect function and the output we append it to the writer that has data about output video
    new_frame = detect_object(frame, neural_net_object.eval(), transform)
    # neural_net_object.eval() is done to convert the neural_net_object to the type that is expected by the detect_object function from 
    # which we can get output y. This is just done to align with the way the build ssd function was made.
    writer.append_data(new_frame)
    print(i)

writer.close()
 
    
    