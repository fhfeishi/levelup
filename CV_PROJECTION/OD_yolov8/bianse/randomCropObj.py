import random

def randomCropBbox(opencvImage, bboxes, crop_size=(640,640), cropBbox_ratio=0.7, max_attempts=100):
    h,w,_ = opencvImage.shape
    cw,ch = crop_size
    assert w>cw and h>ch, "crop < image !!!"
    
    for _ in range(max_attempts):
        x1 = random.randint(0, (w-cw+1))
        y1 = random.randint(0, (h-ch+1))
        x2 = x1 + cw
        y2 = y1 + ch
        
        new_bboxes = []
        for bbox in bboxes:
            cropBbox_x1 = max(x1, bbox[0])
            cropBbox_y1 = max(y1, bbox[1])
            cropBbox_x2 = min(x2, bbox[2])
            cropBbox_y2 = min(y2, bbox[3])
            cropBbox_area = (cropBbox_x2-cropBbox_x1)*(cropBbox_y2-cropBbox_y1)
            bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

            if cropBbox_area >= bbox_area*cropBbox_ratio:
                new_bboxes.append([cropBbox_x1-x1, cropBbox_y1-y1, cropBbox_x2-x1, cropBbox_y2-y2])

        
        if len(new_bboxes)>0:
            croppedImage = opencvImage[y1:y2, x1:x2]
            return croppedImage, new_bboxes
    
    return opencvImage, bboxes
            

