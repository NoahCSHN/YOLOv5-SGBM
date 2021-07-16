import cv2
import time,logging,random,os,sys
import numpy as np
from rknn.api.rknn import RKNN
from utils.general import AutoScale,letterbox,timethis,get_new_size,get_max_scale,plot_one_box

"""
yolov5 预测脚本 for rknn
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def load_model0(model_path, npu_id):
    rknn = RKNN()
    devs = rknn.list_devices()
    device_id_dict = {}
    for index, dev_id in enumerate(devs[-1]):
        if dev_id[:2] != 'TS':
            device_id_dict[0] = dev_id
        if dev_id[:2] == 'TS':
            device_id_dict[1] = dev_id
    print('-->loading model : ' + model_path)
    rknn.load_rknn(model_path)
    print('--> Init runtime environment on: ' + device_id_dict[npu_id])
    ret = rknn.init_runtime(device_id=device_id_dict[npu_id])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def load_rknn_model(PATH):
    rknn = RKNN()
    print('--> Loading model')
    ret = rknn.load_rknn(PATH)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


class RKNNDetector:
    count = 0
    def __init__(self, model, wh, masks, anchors, names,draw_box=False):
        RKNNDetector.count += 1
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.names = names
        if isinstance(model, str):
            model = load_rknn_model(model)
        self._rknn = model
        self.draw_box = draw_box

    def _predict(self, img_src, _img, gain, padding, conf_thres=0.4, iou_thres=0.45):
        src_h, src_w = img_src.shape[:2]
        pred_h,pred_w = _img.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        pred_onx = self._rknn.inference(inputs=[_img])
        logging.info('inference time: (%.2fs)',(time.time() - t0))
        # print('inference time: (%.2fs)'%(time.time() - t0))
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = filter_boxes(box, box_confidence, box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return [], []
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        label_list = []
        box_list = []
        score_list = []
        raw_box_list = []
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x_raw = x * pred_w
            y_raw = y * pred_h
            w_raw = w * pred_w
            h_raw = h * pred_h
            x1_raw = max(0,np.floor(x_raw).astype(int))
            y1_raw = max(padding[0],np.floor(y_raw).astype(int))
            x2_raw = min(_img.shape[1],np.floor(x_raw+w_raw+0.5).astype(int))
            y2_raw = min(_img.shape[0]-padding[1],np.floor(y_raw+h_raw+0.5).astype(int))
            raw_box_list.append((x1_raw,y1_raw,x2_raw,y2_raw))
            x *= gain[0]
            y -= (padding[0]/pred_h) 
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            label_list.append(self.names[cl])
            box_list.append((x1, y1, x2, y2))
            score_list.append(score)
            if self.draw_box:
                plot_one_box((x1, y1, x2, y2), img_src, label=self.names[cl])
        return label_list, score_list, box_list , raw_box_list

    def predict_resize(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    # @timethis
    def predict(self,img_raw, img_src, gain, padding, Queue, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        # _img, gain, padding = letterbox(img_src, self.wh)
        Queue.put(self._predict(img_raw, img_src, gain, padding, conf_thres, iou_thres))
        # return self._predict(img_src, _img, gain, padding, conf_thres, iou_thres), _img.shape

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        RKNNDetector.count -= 1
        class_name=self.__class__.__name__
        print (class_name,"release")
        self.close()


if __name__ == '__main__':
    from utils.dataset import loadfiles
    RKNN_MODEL_PATH = "../weights/best.rknn"
    IMG_PATH = '../data/test08'
    SIZE = (640, 640)
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    model = load_rknn_model(RKNN_MODEL_PATH)
    detector = RKNNDetector(model, SIZE, MASKS, ANCHORS, CLASSES,draw_box=True)
    # img = cv2.imread("../data/images/bus.jpg")
    dataset = loadfiles(IMG_PATH,SIZE)
    for path,imgl,imgr,shape,vid_cap in dataset:
        detector.predict(imgl)
        cv2.imshow("res", imgl)
        while (1):
            if cv2.waitKey(1) == ord('q'):
                break
        # cv2.waitKey()
        cv2.destroyAllWindows()
