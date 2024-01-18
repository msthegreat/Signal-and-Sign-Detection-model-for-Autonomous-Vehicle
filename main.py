import light_detector_classifier
import utils
import sign_detector_classifier
import globals
import cv2
import numpy as np
from os import listdir
import argparse

# Returns a traffic lights with their top-left bounding box coordinates.
# Also returns the output image with bounding boxes.
# NO ANNOTATIONS here since it was not asked for in the project doc. Also because everything is so small already ! :(
def detect_and_classify_lights(image, crop=0.5):
    image = cv2.resize(image, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    coords, result = light_detector_classifier.detect_lights(image, crop)

    result = image.copy()
    final_results = []
    for coord in coords:
        coord = np.array(coord)
        area = result[coord[0,1]:coord[1,1], coord[0,0]:coord[1,0]]
        area = cv2.resize(area, (30, 60), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        light = light_detector_classifier.classify_light(area)

        if light != 0:
            threshold = {
                'red-upper': globals.LIGHT_THRESHOLDS['red'][0],
                'red-lower': globals.LIGHT_THRESHOLDS['red'][1],
                'green': globals.LIGHT_THRESHOLDS['green']
            }
            temp = utils.color_threshold(area, threshold)
            final_results.append((globals.light_labels[light], coord)) if np.count_nonzero(temp) >  50 else None

    for (light, coord) in final_results:
        cv2.rectangle(result, coord[0], coord[1], (0, 255, 0), 2)

    return final_results, result

# Returns a dictionary of signs with their top-left bounding box coordinates.
# Also returns the output image with bounding boxes and annotations rendered.
def detect_and_classify_signs(image):
    image = cv2.resize(image, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    coords, result = sign_detector_classifier.detect_sign(image)

    result = image.copy()
    final_results = {
        'stop': [],
        'pedestrianCrossing': [],
        'signalAhead': []
    }
    for coord in coords:
        area = result[coord[0,1]:coord[1,1], coord[0,0]:coord[1,0]]
        area = cv2.resize(area, (48, 48), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        sign = sign_detector_classifier.classify_sign(area)

        if sign != 0:
            temp = utils.color_threshold(area, {globals.signs[sign] : globals.COLOR_THRESHOLDS[globals.signs[sign]]})
            final_results[globals.sign_labels[sign]].append(coord) if np.count_nonzero(temp) > 0.025*temp.size else None

    for sign in globals.sign_labels.values():
        if sign == "unknown":
            continue
        for coord in final_results[sign]:
            cv2.rectangle(result, coord[0], coord[1], (0, 255, 0), 2)
            cv2.rectangle(result, (coord[0][0]-5, coord[0][1]-15), ((coord[1][0]+10, coord[0][1])), (0,0,0), -1)
            cv2.putText(result, globals.text[sign], (coord[0][0]-5, coord[0][1]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.35, color=(0,0,255))
            cv2.rectangle(result, (coord[0][0] - 5, coord[1][1] +2), ((coord[1][0] + 10, coord[1][1]+15)), (0, 0, 0), -1)
            cv2.putText(result, str(coord[0]), (coord[0][0]-5, coord[1][1]+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(0, 0, 255))

    return final_results, result


if __name__ == '__main__':
    '''
    Sign detection - Five sample images used. Can also be used to run on entire LISA sign dataset if needed.
    # Use the following two paths for command line args to verify the report images for traffic signs
    # input_path = "./Resources/Dataset/Signs"
    # output_path = "./output/signs"

    Light detection - Sample images taken from LISA lights dataset. Can also be used to run on entire LISA sign dataset if needed.
    # Use the following two paths for command line args to verify the report images for traffic lights
    # input_path = "./Resources/Dataset/lightss"
    # output_path = "./output/lights"
    
    '''

    parser = argparse.ArgumentParser(description='Runner program to invoke traffic light detection model or sign recognition model. \n'
                                                 'The two models can be combined to run on each frame. \n'
                                                 'But it is not provided as an option to stick to having code that can run on any arbitrary sign or light dataset')
    parser.add_argument('--type', metavar='--t', type=str,
                        help='Type of detection. Use \'light\' to execute traffic light detection, \'sign\' for traffic sign recognition. ')
    parser.add_argument('--input_path', metavar='--i', type=str,
                        help='Path to input directory containing images or frames to process. Subdirectories are not supported. ')
    parser.add_argument('--output_path', metavar='--o', type=str,
                        help='Path to an existing output directory where the processed frames with annotations and bounding boxes are written to.')

    parser.print_help()
    args = vars(parser.parse_args())

    type = args['type']
    input_path = args['input_path']
    output_path = args['output_path']

    if type == "light":
        model = detect_and_classify_lights
    else:
        model = detect_and_classify_signs

    for file in listdir(input_path):
        image = utils.read_image(input_path + "/" + file, cv2.IMREAD_COLOR)
        final_results, result = model(image)
        cv2.imwrite(output_path+"/"+file, result)


    # !!The long run (Optional run to verify the videos generated in ./output/lights and to benchmark accuracy)!!
    # Light detection - Uses a subset of directories from LISA traffic light dataset to generate videos
    # The subset of data used is uploaded to box here: https://gatech.box.com/s/v2260x3i35owmzht1j5cia1pigdge912
    # Before uncommenting and executing this code, please download the dataset from box to "./Resources/Dataset" and provide the right command line args

    # path = "./Resources/Dataset/Lights/dayClip"
    # video = utils.get_video_writer_object("traffic_light_dayClip")
    # write_results = []
    # for file in listdir(path):
    #     image = utils.read_image(path + "/" + file, cv2.IMREAD_COLOR)
    #     final_results, result = detect_and_classify_lights(image, crop=0.48)
    #     for res in final_results:
    #         write_results.append([file, res[-1][0], res[-1][1]])
    #     video.write(result)
    # cv2.destroyAllWindows()
    # video.release()

    # This is not needed for code execution - It's only needed to calculate accuracy results for report
    # utils.write_coords_to_csv(write_results, "./output/lights", "traffic_light_dayClip.csv")

