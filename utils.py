'''
This file defines all helper functions used
'''

import cv2
import numpy as np
import csv


def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_threshold(image, color_thresholds, color_format=cv2.COLOR_BGR2HSV):
    source = image.copy()
    image = cv2.cvtColor(image, color_format)
    mask = np.zeros((source.shape[0], source.shape[1]), dtype=np.uint8)

    # Get individual color masks and combine them
    for (color, thresholds) in color_thresholds.items():
        lower = thresholds[0]
        upper = thresholds[1]
        color_mask = cv2.inRange(image, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)

    # Apply combined mask on source image
    result = cv2.bitwise_and(source, source, mask=mask)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = np.asarray(result, dtype=np.uint8)

    # show_image(result, "Color thresholded")
    return result



def keypoint_match(template, image):
    sift = cv2.SIFT_create()
    src_kp, s_descriptor = sift.detectAndCompute(template, None)
    dst_kp, d_descriptor = sift.detectAndCompute(image, None)
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(s_descriptor, d_descriptor)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[0:10]

    match_image = cv2.drawMatches(template, src_kp, image, dst_kp, matches,None, flags=2)
    return match_image


def threshold_image(image):
    threshold = 100

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    ret, thresh = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
    watershed = cv2.watershed(image, markers)
    return watershed


def read_image(path, mode):
    # mode: IMREAD_COLOR, IMREAD_GRAYSCALE
    image = cv2.imread(path, mode)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image


def auto_correct_images(path, filename, output_dir):
    filename = path + "/" + filename
    labels = {
        'stop': 0,
        'signalAhead': 0,
        'pedestrianCrossing': 0,
    }
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            img_path = path + "/" + row[1] + "/" + row[0]
            image = read_image(img_path, cv2.IMREAD_COLOR)
            cv2.imwrite(output_dir + "/" + row[1] + "/" + row[1] + "_" + str(labels[row[1]]) + ".png", image)
            labels[row[1]] += 1


def extract_signs_from_images(path, filename, output_dir, labels, new_dims):
    filename = path+"/"+filename

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue

            # img_path = path + "/" + row[1] + "/" + row[0]
            img_path = path + "/" + row[0]
            image = read_image(img_path, cv2.IMREAD_GRAYSCALE)
            top_left = (int(row[2]), int(row[3]))
            bottom_right = (int(row[4]), int(row[5]))
            image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            image = cv2.resize(image, new_dims, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            name = output_dir+"/"+row[1]+"/"+row[1]+"_"+str(labels[row[1]])+".png"
            cv2.imwrite(name, image)
            labels[row[1]] += 1


def get_video_writer_object(file_name, output_dir="./output", width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(output_dir+"/"+file_name+".avi", fourcc, 2, (width, height))
    return video


def write_coords_to_csv(results, path, filename, isSign=False):
    with open(path + "/" + filename, 'w') as file:
        writer = csv.writer(file)
        header = ['file', 'sign', 'upper_left_x', 'upper_left_y'] if isSign else ['file', 'upper_left_x', 'upper_left_y']
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

