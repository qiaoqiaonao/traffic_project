# 验证脚本 verify_coco.py
from pycocotools.coco import COCO
import cv2
import random

coco = COCO("dataset/annotations/instances_train.json")

# 随机显示一张图及其标注
img_ids = coco.getImgIds()
img_id = random.choice(img_ids)
img_info = coco.loadImgs(img_id)[0]

img = cv2.imread(f"dataset/images/{img_info['file_name']}")
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

for ann in anns:
    x, y, w, h = ann['bbox']
    cat = coco.loadCats(ann['category_id'])[0]['name']
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
    cv2.putText(img, cat, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("COCO验证", img)
cv2.waitKey(0)
cv2.destroyAllWindows()