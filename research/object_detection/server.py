# Imports
from aiohttp import web
import os.path
import hashlib
import numpy as np
import os
import threading
import sys
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime
from io import BytesIO


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


# 全局变量、“宏定义”声明

#                   Model name	                Speed	COCO mAP	Outputs
# ssd_mobilenet_v1_coco	                        fast	    21	    Boxes
# ssd_inception_v2_coco	                        fast	    24	    Boxes
# rfcn_resnet101_coco	                        medium	    30	    Boxes
# faster_rcnn_resnet101_coco	                medium	    32	    Boxes
# faster_rcnn_inception_resnet_v2_atrous_coco	slow	    37	    Boxes

# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
# MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

max_image_size = 640
confidence_level = 0.6
redirect_html = ''
image_tensor = ''
detection_boxes = ''
detection_scores = ''
detection_classes = ''
num_detections = ''
tf_sess = ''


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        # 关闭下行注释,用tensorboard查看其模型图Graph "tensorboard --logdir logdir"
        # tf.summary.FileWriter("logdir", detection_graph)


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as tf_sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


class Box:
    def __init__(self, box):
        self.ymin = float(box[0])
        self.xmin = float(box[1])
        self.ymax = float(box[2])
        self.xmax = float(box[3])


class Detected:
    def __init__(self, objid, name, score, box):
        self.objid = objid
        self.name = name
        self.score = float(score)
        self.box = Box(box)


# 图片接收、检测、返回结果
async def od_handler(request):
    reader = await request.multipart()
    rcv_file = await reader.next()
    binary = await rcv_file.read()

    # # rename using MD5
    # md5 = hashlib.md5()
    # md5.update(binary)
    # rename = md5.hexdigest() + os.path.splitext(rcv_file.filename)[1]
    # # store
    # with open(os.path.join('cache', rename), 'wb') as fd:
    #     fd.write(binary)

    # open image with PIL(pillow)
    image = Image.open(BytesIO(binary))

    # resize the image
    (im_width, im_height) = image.size
    if im_width >= im_height:
        if im_width > max_image_size:
            new_width = max_image_size
            new_height = int(im_height * (max_image_size/im_width))
            resize_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            resize_image = image
    else:
        if im_height > max_image_size:
            new_height = max_image_size
            new_width = int(im_width * (max_image_size/im_height))
            resize_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            resize_image = image

    print("Thread(%d) start to detect image : %s" % (threading.get_ident(), rcv_file.filename))
    start_time = datetime.now()
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(resize_image)
    npload_time = datetime.now()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = tf_sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    end_time = datetime.now()
    print(rcv_file.filename
          + " numpy array load time " + str(npload_time - start_time)
          + ", detect time " + str(end_time - npload_time))

    # pick up the high score result
    od_results = []
    for x in range(len(scores)):
        for y in range(len(scores[x])):
            score = scores[x][y]

            if score > confidence_level:
                objid = int(classes[x][y])

                if objid in category_index.keys():
                    class_name = category_index[objid]['name']
                else:
                    class_name = 'N/A'

                obj_tmp = Detected(objid, class_name, score, boxes[x][y])
                od_results.append(obj_tmp)
            else:
                break

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image_np)

    # http response
    res_json = json.dumps(od_results, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    return web.Response(text=res_json)


# 网页重定向
async def redirect_handler(request):
    return web.Response(body=redirect_html, content_type='text/html')


with open('static/redirect.html', 'rt', encoding='utf-8') as file:
    redirect_html = file.read()


webserv = web.Application()
webserv.router.add_get('/', redirect_handler)
webserv.router.add_post('/od', od_handler)
webserv.router.add_static('/', './static')
web.run_app(webserv, port=80)
