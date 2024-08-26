import random

def randomCropBbox_raw(opencvImage, bboxes, crop_size=(640,640), cropBbox_ratio=0.7, max_attempts=100):
    h,w,_ = opencvImage.shape
    cw,ch = crop_size
    assert w>cw and h>ch, "crop < image !!!"
    
    for _ in range(max_attempts):
        x1 = random.randint(0, (w-cw+1))
        y1 = random.randint(0, (h-ch+1))
        x2 = x1 + cw
        y2 = y1 + ch
        
        interAreaRatios = {}
        for bbox in bboxes:
            bbox_x1 = max(x1, bbox[0])
            bbox_y1 = max(y1, bbox[1])
            bbox_x2 = min(x2, bbox[2])
            bbox_y2 = min(y2, bbox[3])
            
            cropBbox_area = max(0, bbox_x2-bbox_x1) * max(0, bbox_y2-bbox_y1)
            bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            interAreaRatio = cropBbox_area / bbox_area
            interAreaRatios[tuple(bbox)] = interAreaRatio
            
            # interAreaRatios = dict(sorted(interAreaRatios.items(), key=lambda item: item[1], reverse=True))

            if any(ratio >= cropBbox_ratio for ratio in interAreaRatios.values()):
                if all(ratio >= cropBbox_ratio or ratio == 0 for ratio in interAreaRatios.values()):
                    cropped_image = opencvImage[y1:y2, x1:x2]

                    new_bboxes = []
                    for bbox, ratio in interAreaRatios.items():
                        if ratio > cropBbox_ratio:
                            updated_bbox = [
                                        max(0, bbox[0] - x1),
                                        max(0, bbox[1] - y1),
                                        min(cw, bbox[2] - x1),
                                        min(ch, bbox[3] - y1)
                                    ]
                            new_bboxes.append(updated_bbox)
                return cropped_image, new_bboxes

    return opencvImage, bboxes
            

def randomCropBboxA(opencvImage, bboxes, crop_size=(640,640), cropBbox_ratio=0.7):
    h, w, _ = opencvImage
    cw, ch = crop_size
    assert w>cw and h>ch, "crop < image !!!"

    while True:
        x1 = random.randint(0, (w-cw+1))
        y1 = random.randint(0, (h-ch+1))
        x2 = x1 + cw
        y2 = y1 + ch
        
        interAreaRatios = {}
        for bbox in bboxes:
            bbox_x1 = max(x1, bbox[0])
            bbox_y1 = max(y1, bbox[1])
            bbox_x2 = min(x2, bbox[2])
            bbox_y2 = min(y2, bbox[3])
            
            cropBbox_area = max(0, bbox_x2-bbox_x1) * max(0, bbox_y2-bbox_y1)
            bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            interAreaRatio = cropBbox_area / bbox_area
            interAreaRatios[tuple(bbox)] = interAreaRatio
            
            # interAreaRatios = dict(sorted(interAreaRatios.items(), key=lambda item: item[1], reverse=True))

            if any(ratio >= cropBbox_ratio for ratio in interAreaRatios.values()):
                if all(ratio >= cropBbox_ratio or ratio == 0 for ratio in interAreaRatios.values()):
                    cropped_image = opencvImage[y1:y2, x1:x2]

                    new_bboxes = []
                    for bbox, ratio in interAreaRatios.items():
                        if ratio > cropBbox_ratio:
                            updated_bbox = [
                                        max(0, bbox[0] - x1),
                                        max(0, bbox[1] - y1),
                                        min(cw, bbox[2] - x1),
                                        min(ch, bbox[3] - y1)
                                    ]
                            new_bboxes.append(updated_bbox)
                return cropped_image, new_bboxes
        
                    
def randomCropBboxB(pillowImage, bboxes, crop_size=(640,640), cropBbox_ratio=0.7, max_attempts=100):
    w, h = pillowImage.size
    cw, ch = crop_size
    assert w>cw and h>ch, "crop < image !!!"
    
    for _ in range(max_attempts):
        x1 = random.randint(0, (w-cw+1))
        y1 = random.randint(0, (h-ch+1))
        x2 = x1 + cw
        y2 = y1 + ch
        interAreaRatios = {}
        for bbox in bboxes:
            bbox_x1 = max(x1, bbox[0])
            bbox_y1 = max(y1, bbox[1])
            bbox_x2 = min(x2, bbox[2])
            bbox_y2 = min(y2, bbox[3])
            
            cropBbox_area = max(0, bbox_x2-bbox_x1) * max(0, bbox_y2-bbox_y1)
            bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            interAreaRatio = cropBbox_area / bbox_area
            interAreaRatios[tuple(bbox)] = interAreaRatio
            
            # interAreaRatios = dict(sorted(interAreaRatios.items(), key=lambda item: item[1], reverse=True))

            if any(ratio >= cropBbox_ratio for ratio in interAreaRatios.values()):
                if all(ratio >= cropBbox_ratio or ratio == 0 for ratio in interAreaRatios.values()):
                    cropped_image = pillowImage.crop((x1,y1,x2,y2)) # Image.crop((left,top,right,bottom)) 

                    new_bboxes = []
                    for bbox, ratio in interAreaRatios.items():
                        if ratio > cropBbox_ratio:
                            updated_bbox = [
                                        max(0, bbox[0] - x1),
                                        max(0, bbox[1] - y1),
                                        min(cw, bbox[2] - x1),
                                        min(ch, bbox[3] - y1)
                                    ]
                            new_bboxes.append(updated_bbox)
                return cropped_image, new_bboxes
        
    return pillowImage, bboxes
    

"""
h,w,_ = opencvImage.shape
imageCrop = opencvImage[y1:y2, x1:x2]

w,h = pillowImage.size
imageCrop = pillowImage.crop((x1,y1,x2,y2))

"""


def myCrop(pillowImage, bboxes, crop_size=(640,640), ratio=0.7, maxAttempts=100):
    w,h = pillowImage.size
    
    for _ in range(maxAttempts):
        x1 = random.randint(0, (w-crop_size[0]+1))
        y1 = random.randint(0, (h-crop_size[1]+1))
        x2 = x1 + crop_size[0]
        y2 = y1 = crop_size[1]
        
        interAreaDicts = {}
        for bbox in bboxes:
            interBbox_x1 = max(x1, bbox[0])
            interBbox_y1 = max(x1, bbox[1])
            interBbox_x2 = max(x1, bbox[2])
            interBbox_y2 = max(x1, bbox[3])
            
            interArea = (interBbox_x2-interBbox_x1) * (interBbox_y2-interBbox_y1)
            bboxArea = (bbox[3]-bbox[1]) * (bbox[2]-bbox[1])
            interAreaDicts[tuple(bbox)] = interArea / bboxArea
        
        # interAreaDicts = dict(sorted(interAreaDicts.items, key=lambda item:item[1], reverse=True))
        
        if any(ratio >= ratio for ratio in interAreaDicts.values()) and \
        all(ratio >= ratio or ratio == 0 for ratio in interAreaDicts.values()):
            
            croppedImage = pillowImage.crop((x1, y1, x2, y2))
            
            new_bboxes = []
            for key, value in interAreaDicts:
                if value >= ratio:
                    new_bbox = {
                        max(0,key[0]-x1),
                        max(0,key[1]-y1),
                        min(crop_size[0],key[2]-x2),
                        min(crop_size[1],key[3]-y2),
                    }
                    new_bboxes.append(new_bbox)
            
            return croppedImage, new_bboxes
        
    return pillowImage, bboxes