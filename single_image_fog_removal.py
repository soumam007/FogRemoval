import numpy as np
import cv2
import math
import timeit

start=timeit.default_timer()                      #Starts the timer to estimate the full processing time

img=cv2.imread('./img/fog67.jpg',1)
b,g,r =cv2.split(img)                             #splits the image into its 3 channels
h= r.shape[0]
w= r.shape[1]
print("height--->",h)
print("Width---->",w)

atmospheric_light=[0,0,0]                           #initializing atmospheric_light
dark_channel=np.zeros(shape=(h,w),dtype='uint8')    #declaring dark_channel
transmission=np.zeros(shape=(h,w),dtype='f8')       # declaring transmission

start_darkChannelAndTransmission_estimation=timeit.default_timer()      #Starts the timer to estimate the dark_channeland transmission time

start_dk_channel=timeit.default_timer()
dark_channel=np.minimum(np.minimum(b,g),r)
stop_dk_channel=timeit.default_timer()
print("dark channel estimate========>",stop_dk_channel-start_dk_channel)

start_atm_light=timeit.default_timer()
indexMax=np.unravel_index(np.argmax(dark_channel),(h,w))
index= (indexMax[1],indexMax[0])                                # gets the index of the brightest pixel
atmospheric_light=img[indexMax[0]][indexMax[1]]                 # get the channel values of r,g,b of the brightest pixel(i.e. index)
print("index of the image where we get the atmospheric light from--->",index)
print ("atmospheric_light------>",atmospheric_light)
stop_atm_light=timeit.default_timer()
print("atm light estimate========>",stop_atm_light-start_atm_light)



start_transmission=timeit.default_timer()
b_by_atmLightB=np.zeros(shape=(h,w),dtype='f8')
b_by_atmLightB=b/atmospheric_light[0]
g_by_atmLightG=np.zeros(shape=(h,w),dtype='f8')
g_by_atmLightG=g/atmospheric_light[1]
r_by_atmLightR=np.zeros(shape=(h,w),dtype='f8')
r_by_atmLightR=r/atmospheric_light[2]

minimum_of_the_above_3=np.zeros(shape=(h,w),dtype='f8')
minimum_of_the_above_3=np.minimum(np.minimum(b_by_atmLightB,g_by_atmLightG),r_by_atmLightR)

transmission=np.ones(shape=(h,w))-minimum_of_the_above_3
stop_transmission=timeit.default_timer()
print("transmission estimate========>",stop_transmission-start_transmission)



stop_darkChannelAndTransmission_estimation=timeit.default_timer()   #Stops the timer to estimate the dark_channeland transmission time

print("time taken to estimate dark_channel and transmission----->",stop_darkChannelAndTransmission_estimation-start_darkChannelAndTransmission_estimation)


r_restored_image=np.zeros(shape=(h,w),dtype='uint8')
g_restored_image=np.zeros(shape=(h,w),dtype='uint8')
b_restored_image=np.zeros(shape=(h,w),dtype='uint8')
to=0.6                           # its generally taken 0.1 but I experimetally found that its giving better result for t=0.6


start_restoring=timeit.default_timer()        # Starts the timer to estimate the restoring time
division_matrix=np.zeros(shape=(h,w),dtype='f8')
to_matrix=np.full((h, w), 0.6)
division_matrix=np.maximum(transmission,to_matrix)


atmospheric_light_blue=np.zeros(shape=(h,w),dtype='uint8')
atmospheric_light_green=np.zeros(shape=(h,w),dtype='uint8')
atmospheric_light_red=np.zeros(shape=(h,w),dtype='uint8')

atmospheric_light_blue=np.full((h, w), atmospheric_light[0])
atmospheric_light_green=np.full((h, w), atmospheric_light[1])
atmospheric_light_red=np.full((h, w), atmospheric_light[2])

b_restored_image=((b.astype('int')-atmospheric_light_blue.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_blue.astype('uint8')
g_restored_image=((g.astype('int')-atmospheric_light_green.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_green.astype('uint8')
r_restored_image=((r.astype('int')-atmospheric_light_red.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_red.astype('uint8')


restored_image=cv2.merge((b_restored_image,g_restored_image,r_restored_image))
stop_restoring=timeit.default_timer()    #Stops the timer to estimate the restoring time

print("time taken to restore the image--->",stop_restoring-start_restoring)

start_display=timeit.default_timer()   #Starts the timer to estimate the displaying time
titles=['input image','dark_channel','transmission','b_restored_image','g_restored_image','r_restored_image','restored_image']
images=[cv2.merge((b,g,r)),dark_channel,transmission,b_restored_image,g_restored_image,r_restored_image,restored_image]#,dark_channel]



for i in range(len(titles)):
    cv2.namedWindow(titles[i],cv2.WINDOW_NORMAL)
    cv2.imshow(titles[i],images[i])
    i=i+1

stop_display=timeit.default_timer()

print("time to display---->",stop_display-start_display)  #Stops the timer to estimate the displaying time

stop=timeit.default_timer()                  #Stops the timer to estimate the full processing time

print("the time taken to do the full processing is---->",stop-start," secs")

k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
