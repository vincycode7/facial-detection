#import libraries
import torch
import torch.onnx
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
import numpy as np
from imutils.video import VideoStream,FPS
import time
from data_load import *
from models import *

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

#load model
model_struct = torch.load('./saved_models/keypoints_model_frt_tst10.pth')
model = model_struct['arc']
model.load_state_dict(model_struct['state_dict'])

#convert to onnx
sample_batch_size, channel, height, width = 1,1,224,224
dummy_input = torch.randn(sample_batch_size, channel, height, width)
model.eval()
# data_transform = transforms.Compose([Rescale(250),
#                                      RandomCrop(224),
#                                      Normalize(),
#                                      ToTensor()])
#torch.onnx.export(model, dummy_input, 'face_keypts.onnx')

#input and output
#vid = cv2.VideoCapture(0)
# while True:
#     ret, frame = vid.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frm_cp = np.copy(frame)
#     #pilimg = Image.fromarray(frame)
#     frm_cp = cv2.resize(frm_cp,(224,224))
#     print(frame.shape)
#     break
#     # plt.imshow(frame)

#     fig=plt.figure(figsize=(12, 8))
#     plt.title("Video Stream")
#     plt.imshow(frame)
#     plt.show()
#     clear_output(wait=True)

#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
vs.set(3,100)
vs.set(4,100)
time.sleep(2.0)
fps = FPS().start()

while True:
    flag, frame = vs.read()

    if not flag:
        break

    frm_cp = np.copy(frame)
    frm_cp = cv2.resize(frm_cp,(224,224))
    img = cv2.cvtColor(frm_cp, cv2.COLOR_BGR2GRAY)/255

    #reshape
    img = img.reshape(1,1,224,224)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    output_pts = model(img)
    fac_key = output_pts.view(output_pts.size()[0], 68, -1)[0].data.numpy()*50+100
    #plot image and keypts
    for x,y in fac_key:
        cv2.circle(frm_cp,(x,y),2,(0,0,255),-1,)

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    #create a window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # resize window
    cv2.resizeWindow("Frame", 600,600)
    #display image on window
    cv2.imshow("Frame",frm_cp)
    out.write(frame)


    #wait for key to stop the program
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
	# update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
out.release()
vs.release()
cv2.destroyAllWindows()
#vs.stop()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
out.release()
vs.release()