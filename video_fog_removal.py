import numpy as np
import cv2
import math
import time
import timeit



start=timeit.default_timer()

cap = cv2.VideoCapture('fog3.mp4')

print (cap.get(3))
print (cap.get(4))
w=int(cap.get(3))
h=int(cap.get(4))



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_fog2.mp4',fourcc, 14 , (w,h))




dark_channel=np.zeros(shape=(h,w),dtype='uint8')
transmission=np.zeros(shape=(h,w),dtype='f8')

r_restored_image=np.zeros(shape=(h,w),dtype='uint8')
g_restored_image=np.zeros(shape=(h,w),dtype='uint8')
b_restored_image=np.zeros(shape=(h,w),dtype='uint8')


b_by_atmLightB=np.zeros(shape=(h,w),dtype='f8')
g_by_atmLightG=np.zeros(shape=(h,w),dtype='f8')
r_by_atmLightR=np.zeros(shape=(h,w),dtype='f8')


minimum_of_the_above_3=np.zeros(shape=(h,w),dtype='f8')

r_restored_image=np.zeros(shape=(h,w),dtype='uint8')
g_restored_image=np.zeros(shape=(h,w),dtype='uint8')
b_restored_image=np.zeros(shape=(h,w),dtype='uint8')


atmospheric_light_blue=np.zeros(shape=(h,w),dtype='uint8')
atmospheric_light_green=np.zeros(shape=(h,w),dtype='uint8')
atmospheric_light_red=np.zeros(shape=(h,w),dtype='uint8')


division_matrix=np.zeros(shape=(h,w),dtype='f8')
z=0
while (True):

    font = cv2.FONT_HERSHEY_SIMPLEX

    #time.sleep(0.06)
    z=z+1
    print(z)

    ret, frame= cap.read()
    b,g,r =cv2.split(frame)


    cv2.putText(frame,'Frame no: '+str(z),(30,30), font, 1,(0,0,255),2,cv2.LINE_AA)

    dark_channel=np.minimum(np.minimum(b,g),r)

    indexMax=np.unravel_index(np.argmax(dark_channel),(h,w))
    index=  (indexMax[1],indexMax[0])
    atmospheric_light=frame[indexMax[0]][indexMax[1]]
    print(index)
    print (atmospheric_light)

    start_transmission=timeit.default_timer()
    b_by_atmLightB=b/atmospheric_light[0]
    g_by_atmLightG=g/atmospheric_light[1]
    r_by_atmLightR=r/atmospheric_light[2]

    minimum_of_the_above_3=np.minimum(np.minimum(b_by_atmLightB,g_by_atmLightG),r_by_atmLightR)
    transmission=np.ones(shape=(h,w))-minimum_of_the_above_3


    to=0.6

    start_restoring=timeit.default_timer()
    to_matrix=np.full((h, w), to)
    division_matrix=np.maximum(transmission,to_matrix)

    atmospheric_light_blue=np.full((h, w), atmospheric_light[0])
    atmospheric_light_green=np.full((h, w), atmospheric_light[1])
    atmospheric_light_red=np.full((h, w), atmospheric_light[2])


    b_restored_image=((b.astype('int')-atmospheric_light_blue.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_blue.astype('uint8')
    g_restored_image=((g.astype('int')-atmospheric_light_green.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_green.astype('uint8')
    r_restored_image=((r.astype('int')-atmospheric_light_red.astype('int'))/division_matrix.astype('f8')).astype('uint8')+atmospheric_light_red.astype('uint8')


    restored_image=cv2.merge((b_restored_image,g_restored_image,r_restored_image))
    #write the restored frame
    out.write(restored_image)

    #cv2.putText(restored_image,'Frame no: '+str(z),(30,30), font, 1,(0,0,255),2,cv2.LINE_AA)

    titles=['input ','output']
    images=[frame,restored_image]



    for i in range(len(titles)):
        cv2.namedWindow(titles[i],cv2.WINDOW_NORMAL)
        cv2.imshow(titles[i],images[i])
    stop=timeit.default_timer()
    print("the time taken to do the full processing is---->",stop-start," secs")

    k=cv2.waitKey(1)

    if k==27:
        cv2.destroyAllWindows()
        out.release()
        cap.release()

    elif k== ord('q'):
        cv2.destroyAllWindows()
        out.release
        cap.release()
