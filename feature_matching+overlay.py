import cv2
import numpy as np

def get_topdown_quad(image, src):
    # src and dst points
    src = order_points(src)

    (max_width, max_height), max_range = max_width_height(src)
    dst = topdown_points(max_width, max_height)

    # warp perspective
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width,max_height))

    # return top-down quad
    return warped

def order_points(points):
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    ordered_points = np.zeros((4, 2), dtype="float32")

    ordered_points[0] = points[np.argmin(s)]
    ordered_points[2] = points[np.argmax(s)]
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[3] = points[np.argmax(diff)]

    return ordered_points

def max_width_height(points):
    (tl, tr, br, bl) = points
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(top_width), int(bottom_width))

    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    max_range = int(np.sqrt(max_width **2 + max_height **2))

    return (max_width, max_height), max_range

def topdown_points(max_width, max_height):
    return np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

img = cv2.imread("../reference/model1.jpg", cv2.IMREAD_GRAYSCALE) # queryiamge
cap = cv2.VideoCapture(0)

# Features
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # trainimage
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    #matches = flann.knnMatch(np.asarray(desc_image, np.float32), np.asarray(desc_grayframe, np.float32), k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    # Homography
    if len(good_points) > 15:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

        # Perspective transform
        if not (matrix is None) :
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            matching_top_point = tuple(dst[np.argmin([x[0][1] for x in dst])][0])

            #tl = (int(dst[0][0][0]), int(dst[0][0][1]))
            #tr = (int(dst[3][0][0]), int(dst[3][0][1]))
            #bl = (int(dst[1][0][0]), int(dst[1][0][1]))
            #br = (int(dst[2][0][0]), int(dst[2][0][1]))

            pos = order_points(dst.reshape(4, 2))
            (tl, tr, br, bl) = pos

            min_x = min(int(tl[0]), int(bl[0]))
            min_y = min(int(tl[1]), int(tr[1]))


            for point in pos:
                point[0] = point[0] - min_x
                point[1] = point[1] - min_y

            (max_width, max_height), max_range = max_width_height(pos)
            src = topdown_points(max_width, max_height)

            # warp perspective (with white border)
            overlay = cv2.imread("../reference/overlay1.jpg")

            if (max_width != 0) & (max_height != 0) :
                overlay = cv2.resize(overlay, (max_width, max_height))

                #warped = np.zeros((max_range, max_range, 3), np.uint8)
                warped = frame[min_y:min_y + max_range, min_x:min_x + max_range]
                M = cv2.getPerspectiveTransform(src, pos)
                cv2.warpPerspective(overlay, M, (max_range, max_range), warped, borderMode=cv2.BORDER_TRANSPARENT)

                # add substitute quad
                frame[min_y:min_y + max_range, min_x:min_x + max_range] = warped

                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                #cv2.circle(homography, (int(dst[0][0][0]),int(dst[0][0][1])),3,(255,0,0),5)
                #cv2.circle(homography, (int(dst[1][0][0]), int(dst[1][0][1])), 3, (0, 255, 0), 5)
                #cv2.circle(homography, (int(dst[2][0][0]), int(dst[2][0][1])), 3, (0, 0, 255), 5)
                cv2.putText(homography, "model1", matching_top_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                cv2.imshow("Homography", homography)

    else:
        cv2.imshow("Homography", grayframe)
    #cv2.imshow("Image", img)
    #cv2.imshow("grayFrame", grayframe)
    cv2.imshow("img3", img3)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()