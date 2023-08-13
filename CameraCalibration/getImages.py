import cv2

CAMERA_WIDTH = 1280 # 1920 1280 640
CAMERA_HEIGHT = 720 # 1080 720 360
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)
 
    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        print('clicked')
        route = f'images_{CAMERA_WIDTH}_{CAMERA_HEIGHT}'
        cv2.imwrite(route + '/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()