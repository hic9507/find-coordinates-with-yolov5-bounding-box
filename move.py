import cv2, sys
import copy
import numpy as np
from line_detector import detector


def point_gen(M, N, space, num_of_pixel, shape):
    x1, y1 = M[0][0], M[0][1]
    x2, y2 = N[0][0], N[0][1]

    for i in range(1, num_of_pixel + 1):
        if ((x1 != x2) and (y1 == y2)):
            M.insert(0, [x1, y1 - space * i])
            M.append([x1, y1 + space * i])
            N.insert(0, [x2, y2 - space * i])
            N.append([x2, y2 + space * i])

        if ((x1 == x2) and (y1 != y2)):
            M.insert(0, [x1 - space * i, y1])
            M.append([x1 + space * i, y1])
            N.insert(0, [x2 - space * i, y2])
            N.append([x2 + space * i, y2])

        if M[0][1] < 0:  ####왼쪽 튀어나감
            del M[0]

        if M[int(len(M) - 1)][0] > shape[1]:  ####오른쪽 튀어나감
            del M[int(len(M) - 1)]

        if M[0][0] < 0:  ####위쪽 튀어나감
            del M[0]

        if M[int(len(M) - 1)][1] > shape[0]:  ####아랫쪽 튀어나감
            del M[int(len(M) - 1)]

        if N[0][1] < 0:  ####왼쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][0] > shape[1]:  ####오른쪽 튀어나감
            del N[int(len(N) - 1)]

        if N[0][0] < 0:  ####위쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][1] > shape[0]:  ####아랫쪽 튀어나감
            del N[int(len(N) - 1)]

    return M, N


image = cv2.imread('nonbbox.jpg')
image_gray = cv2.imread('nonbbox.jpg', cv2.IMREAD_GRAYSCALE)

#### 검은 배경에 사각형
image_gray = np.zeros(image_gray.shape, np.uint8)
image = np.zeros(image.shape)
#
space_test = 20 * 3
cv2.line(image_gray, (81, 63 + space_test), (1035, 63), (255, 255, 255), 1)
cv2.line(image_gray, (81, 63 + space_test), (81, 679), (255, 255, 255), 1)
cv2.line(image_gray, (81, 679), (1035, 679), (255, 255, 255), 1)
cv2.line(image_gray, (1035, 63), (1035, 679), (255, 255, 255), 1)

cv2.line(image, (81, 63 + space_test), (1035, 63), (255, 255, 255), 1)
cv2.line(image, (81, 63 + space_test), (81, 679), (255, 255, 255), 1)
cv2.line(image, (81, 679), (1035, 679), (255, 255, 255), 1)
cv2.line(image, (1035, 63), (1035, 679), (255, 255, 255), 1)

# 직선의 방정식 equation of a straight line
x1, y1, x2, y2 = 81, 63, 1035, 63
x3, y3, x4, y4 = 81, 679, 1035, 679

# 이럴까봐 이걸 변수화 해둔거임
space = 10
num_of_pixel = 5  ##정확히는, 기준 픽셀 위로, 아래로 각각 num_of_pixel개 생성됨

A = [[x1, y1]]
B = [[x2, y2]]
C = [[x3, y3]]
D = [[x4, y4]]

# A, C = point_gen(A, C)

Z, X = point_gen(copy.deepcopy(A), copy.deepcopy(B), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(A), copy.deepcopy(C), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(B), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(C), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))