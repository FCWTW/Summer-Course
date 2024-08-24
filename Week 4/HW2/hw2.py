import cv2
import numpy as np
import os

#   set your images' path
path1 = 'Week 4/HW2/image/matching1.jpg'    # Right
path2 = 'Week 4/HW2/image/matching2.jpg'    # Left


def imageShow(string, *args):

    if len(args)>1:
        #   make sure all images have the same height
        heights = [image.shape[0] for image in args]
        min_height = min(heights)

        resized_images = []
        for image in args:
            # change channels of images to 3
            if len(image.shape) == 2:  # image is gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            resized_image = cv2.resize(image, (int(image.shape[1] * min_height / image.shape[0]), min_height))
            resized_images.append(resized_image)

        #   combine images
        combined_image = np.hstack(resized_images)
    else:
        combined_image = args[0]

    cv2.imshow(string, combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combined_image


if __name__ == '__main__':

    #   read images
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    directory = os.path.dirname(path1)

    #   convert the image to Grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)


    #   find feature points with SIFT
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)

    #   result of Feature Left and Feature Right
    image1_point = cv2.drawKeypoints(image1_gray, keypoints1, image1_gray)
    image2_point = cv2.drawKeypoints(image2_gray, keypoints2, image2_gray)

    result = imageShow("Feature Right", image1, image1_point)
    cv2.imwrite(os.path.join(directory+'/right_feature.jpg'), result)

    result = imageShow("Feature Left", image2, image2_point)
    cv2.imwrite(os.path.join(directory+'/left_feature.jpg'), result)


    #   matching features with knn match
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    print(matches[0])

    #   apply ratio test to select good matches
    good_matches = []
    for m in matches:
        if m[0].distance < 0.75 * m[1].distance:
            good_matches.append(m)
    print(good_matches[0])
    image_match = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    result = imageShow("Feature Matching", image_match)
    cv2.imwrite(os.path.join(directory+'/feature_matching.jpg'), result)

    # Extract matched keypoints
    matches = np.asarray(good_matches)
    if len(matches[:,0])>=4:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

    #   Compute homography matrix 𝐻
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)

    #   Perspective Transformation
    width = image1.shape[1] + image2.shape[1]
    height = max(image1.shape[0], image2.shape[0])
    result_image = cv2.warpPerspective(image1, H, (width, height))
    # imageShow("Perspective Transformation", result_image)

    #   Combine images
    result_image[0:image2.shape[0], 0:image2.shape[1]] = image2

    result = imageShow("Result", result_image)
    cv2.imwrite(os.path.join(directory+'/result.jpg'), result)
    