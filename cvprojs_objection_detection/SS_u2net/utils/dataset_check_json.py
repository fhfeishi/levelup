import os
import json
from tqdm import tqdm 

def get_ss_classes(json_folder, stamp=True):
    classes = set()
    for file in tqdm(os.listdir(json_folder)):
        path = os.path.join(json_folder, file)
        if os.path.isfile(path) and path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "shapes" in data:
                    for shape in data["shapes"]:
                        classes.add(shape['label'])
    if stamp:
        print(classes)
    return classes

def remove_noMatched_json(json_name_set, train_name_set, test_name_set, json_folder):
    names_needRemove = json_name_set - train_name_set - test_name_set

    for name in tqdm(names_needRemove):
        path = os.path.join(json_folder, name + '.json')
        if os.path.isfile(path):
            os.remove(path)
            print(f"remove '{name}'.json")
        else:
            print(f"'{name}' no filename, why?")
   
if __name__ == '__main__':

    train_img_folder = r"CV_PROJECTION\SOD_u2net\raw_data\DUTS-TR\DUTS-TR-Image"
    train_name_set = {name.replace('.jpg', '') for name in os.listdir(train_img_folder)}
    print("len train data", len(train_name_set))

    test_img_folder = r"CV_PROJECTION\SOD_u2net\raw_data\DUTS-TE\DUTS-TE-Image"
    test_name_set = {name.replace('.jpg', '') for name in os.listdir(test_img_folder)}
    print('len test data', len(test_name_set))

    json_folder = r"CV_PROJECTION\SOD_u2net\raw_data\json"
    json_name_set = {name.replace('.json', '') for name in os.listdir(json_folder)}
    print("len json data", len(json_name_set))


    no_zb_fname_set = json_name_set - train_name_set - test_name_set

    print("len dataset-ou data", len(no_zb_fname_set)) # 415


    # classes = get_ss_classes(json_folder) # {'jyz', 'zb'}

    remove_noMatched_json(json_name_set, train_name_set, test_name_set, json_folder)




