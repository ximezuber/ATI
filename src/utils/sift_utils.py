import cv2 as cv

def key_points(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv.drawKeypoints(gray,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img

def sift(filename_1: str, filename_2: str, threshold, m = 10, keypoints_1=None, descriptors_1=None):
    img_1 = cv.imread(filename_1)
    img_2 = cv.imread(filename_2)

    img_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    if keypoints_1 == None:
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2,None)

    #feature matching
    bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)

    matches = bf.match(descriptors_1, descriptors_2)
    total_matches_count = len(matches)
    valid_matches = []
    for match in matches:
        if match.distance <= threshold:
            valid_matches.append(match)
    valid_matches_count = len(valid_matches)
    valid_matches = sorted(valid_matches, key = lambda x:x.distance)

    max_matches = min(valid_matches_count, m)

    img_3 = cv.drawMatches(img_1, keypoints_1, img_2, keypoints_2, valid_matches[:max_matches], img_2, flags=2)

    return img_3, valid_matches_count/total_matches_count, keypoints_1, descriptors_1


