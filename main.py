from new_watermark import remove_watermark
import cv2

path='D:/lee/yolo_tensorflow-master/data/pascal_voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
remover=remove_watermark()
remover.build_graph()

result,flag=remover.add_watermark(path)
cv2.imshow('result',result)
print(flag)
cv2.waitKey(0)


