import cv2
import time
import random
import numpy as np
import onnxruntime
from utils.dataset import loadfiles


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    print("boxes shape:             \t", boxes.shape)
    print("box_confidences.shape:    \t", box_confidences.shape)
    print("box_class_probs.shape:   \t", box_class_probs.shape)
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    print("box_class_scores.shape:  \t", box_class_scores.shape)
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    print("pos.shape:               \t", (len(pos),) + pos[0].shape)
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


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    new_img, scale = auto_resize(img, *new_wh)
    shape = new_img.shape
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - shape[0], 0, new_wh[0] - shape[1], cv2.BORDER_CONSTANT,
                                 value=color)
    return new_img, (new_wh[0] / scale, new_wh[1] / scale)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def load_model(path):
    session = onnxruntime.InferenceSession(path)
    input_names = list(map(lambda x: x.name, session.get_inputs()))
    output_names = list(map(lambda x: x.name, session.get_outputs()))
    return session, input_names, output_names


class ONNXDetector:
    def __init__(self, model, wh, masks, anchors, classes):
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.classes = classes
        self.sess, self.input_names, self.output_names = load_model(model)
        self.draw_box = False

    def _predict(self, img_src, _img, gain, conf_thres=0.4, iou_thres=0.45):
        src_h, src_w = img_src.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _img = _img.transpose(2, 0, 1)  # to 3x416x416
        _img = _img[None]
        # img = np.concatenate((img[..., ::2, ::2], img[..., 1::2, ::2], img[..., ::2, 1::2], img[..., 1::2, 1::2]), 1)
        _img = _img.astype(np.float32)
        _img /= 255.0
        t0 = time.time()
        pred_onx = self.sess.run(self.output_names, {self.input_names[0]: _img})
        print("inference time:\t", time.time() - t0)
        boxes, classes, scores = [], [], []
        for t in range(3):
            out = sigmoid(pred_onx[t][0])  # 直接sigmoid处理
            print("out shape", out.shape)
            out = np.transpose(out, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = out.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = out[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = out[..., 5:]
            box_xy = out[..., :2]
            box_wh = out[..., 2:4]
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
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            label_list.append(self.classes[cl])
            box_list.append((x1, y1, x2, y2))
            if self.draw_box:
                plot_one_box((x1, y1, x2, y2), img_src, label=self.classes[cl])
        return label_list, box_list

    def predict_resize(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    def predict(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain, conf_thres, iou_thres)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    # MODEL = r"best_416x256.onnx"
    # SIZE = (416, 256)
    IMG_PATH = './data/test08'
    MODEL = "./weights/best_640x640.onnx"
    SIZE = (640, 640)
    MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    # ANCHORS = [[5, 6], [11, 10], [8, 22], [19, 20], [16, 41], [33, 39], [32, 97], [74, 147], [166, 96]]
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    detector = ONNXDetector(MODEL, SIZE, MASKS, ANCHORS, CLASSES)
    detector.draw_box = True
    # img = cv2.imread("./data/images/bus.jpg")
    # t0 = time.time()
    dataset = loadfiles(IMG_PATH,SIZE)
    for path,imgl,imgr,shape,vid_cap in dataset:
        detector.predict(imgl)
        cv2.imshow("res", imgl)
        while (1):
            if cv2.waitKey(1) == ord('q'):
                break
        # cv2.waitKey()
        cv2.destroyAllWindows()
