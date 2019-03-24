import cv2
import numpy as np
import time
from socket import *
import math

hsv_min = np.array((5, 165, 110), np.uint8)
hsv_max = np.array((20, 255, 205), np.uint8)
mark_hsv_min = np.array((95, 75, 140), np.uint8)
mark_hsv_max = np.array((100, 170, 250), np.uint8)
HReal = 500
HCamera = 1000
SCamera = 1000

def MoveRobot(x,y,z):

    IP = "192.168.0.1"  # IP Robot 192.168.0.1
    PORT = 6610
    # print ("Test 1")
    addr = (IP, PORT)
    sc = socket(AF_INET, SOCK_DGRAM)

    eq = 0
    # максимальный радиус 225 мм
    translateX = x
    translateY = y
    translateZ = z  # максимальная высота 0, минимальная -334
    i = 0
    while True:
        # sc.accept()
        str1 = str(translateX) + " " + str(translateY) + " " + str(translateZ)
        sc.sendto(str1.encode("utf-8"), addr)
        # eq = eq + 1
        if(i>20):
            break


def FindMarker(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, mark_hsv_min, mark_hsv_max)
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    x = 0; y = 0
    if dArea > 100:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
    return x, y

def PrintMarker(x, y, img):
    cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
    cv2.putText(img, "%d-%d" % (x, y), (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def FindCenter(m):
    x = (m[2][0]-m[3][0])/2+m[3][0]
    y = int(((x - m[3][0]) * (m[1][1] - m[3][1])) / (m[1][0] - m[3][0]) + m[3][1])
    return int(x), y

def Trap2Square(y0, x0, z0, top, bottom, HPixel, CenterX): #Подаем (x,y,z)
    try:
        # По X
        KX = (HPixel - CenterX) / CenterX
        x1 = HReal * KX / ((HPixel / x0) - 1 + KX) + CenterX - 440
        # По Y
        h = x0
        x = y0
        y1pix = y0 + x0 * (bottom-top)/(2*HPixel)
        y1 = y1pix*(HReal/bottom)
        # По Z
        l = math.sqrt(HCamera**2+(SCamera+x1)**2)-z0
        z1 = (int)(l/z0*HCamera)

    except(Exception):
        x1 = 0
        y1 = 0
        z1 = 0
        print("error")

    x1+=11
    if (x>0):
        x*=1.13
    y1-=35
    y1*=1.3
    return x1,y1,z1

# Калибровка
cap = cv2.VideoCapture(0)
flag, img = cap.read()
time.sleep(2)
flag, img = cap.read()
img = img[50:len(img)-20, 0:len(img[0])]
# Разбиение на части для нахождения меток
high = len(img); width = len(img[0])
img1 = img[0:high // 2, 0:width // 2]
img2 = img[0:high // 2, width // 2:width]
img3 = img[high // 2:high, width // 2:width]
img4 = img[high // 2:high, 0:width // 2]
# Нахождение координат центров маркеров
x1, y1 = FindMarker(img1)
x2, y2 = FindMarker(img2)
x3, y3 = FindMarker(img3)
x4, y4 = FindMarker(img4)
x2 += width // 2
y3+= high // 2; x3 += width // 2
y4 += high // 2
xCenter, yCenter = FindCenter([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
i = 0
xs=0
ys=0
zs=0
xmove = 0
ymove = 0
zmove = 0
while cap.isOpened():
    flag, img = cap.read()
    img = img[50:len(img)-20, 0:len(img[0])]
    PrintMarker(x1, y1, img)
    PrintMarker(x2, y2, img)
    PrintMarker(x3, y3, img)
    PrintMarker(x4, y4, img)
    PrintMarker(xCenter, yCenter, img)
    # преобразуем RGB картинку в HSV модель
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # применяем цветовой фильтр
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    # вычисляем моменты изображения
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    # будем реагировать только на те моменты,
    # которые содержать больше 100 пикселей
    xRes = 0; yRes = 0
    if dArea > 100:
        xRes = int(dM10 / dArea)
        yRes = int(dM01 / dArea)
        PrintMarker(xRes, yRes, img)
    xt, yt, zt = Trap2Square(xCenter - xRes, y4 - yRes, 1, x2 - x1, x3 - x4, y4 - y1, y4 - yCenter)
    xs+=xt
    ys+=yt
    zs+=zt
    n = 15
    if (i%n==0):
        xmove = xs//n
        ymove = ys//n
        zmove = zs//n
        print('Координаты x,y: "%d %d %d"' % (xmove, ymove, -330))
        xs=0
        ys=0
        zs=0
    i+=1


    cv2.imshow('result', img)
    ch = cv2.waitKey(5)
    if ch == 112:
        input()
    if ch == 109:
        MoveRobot(xmove, ymove, zmove)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()