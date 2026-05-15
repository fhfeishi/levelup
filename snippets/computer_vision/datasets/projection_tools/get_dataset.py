import os
import shutil
from tqdm import tqdm 
import glob

def moveXml_byjpg(jpg_dir, xml_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for file in tqdm(os.listdir(jpg_dir)):
        if file.endswith('.jpg'):
            xmlfile = file.replace('.jpg', '.xml')
            xmlpath = f"{xml_dir}/{xmlfile}"
            xmlpath_save = f"{save_dir}/{xmlfile}"
            shutil.copyfile(xmlpath, xmlpath_save)
                    
    


def moveJpg_byxml(jpg_dir, xml_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for file in tqdm(os.listdir(xml_dir)):
        if file.endswith('.xml'):
            jpgfile = file.replace('.xml', '.jpg')
            # jpgpath = f"{jpg_dir}/{jpgfile}"
            jpgpath_list = glob.glob(os.path.join(jpg_dir, '**', jpgfile), recursive=True)
            if jpgpath_list:
                # Assuming only one match is found, take the first result
                jpgpath = jpgpath_list[0]
                jpgpath_save = os.path.join(save_dir, jpgfile)
                shutil.copyfile(jpgpath, jpgpath_save)
            else:
                # No matching file found
                print(f"No JPEG file found for {file}")


# # move jpgs
# jpgs_dir = r"D:\ddesktop\monitoring\framesdata\frames_"
# xmls_dir = r"D:\ddesktop\monitoring\framesdata\xmls_"
# save_dir = r"D:\ddesktop\monitoring\framesdata\dataset\jpgs"
# moveJpg_byxml(jpg_dir=jpgs_dir, xml_dir=xmls_dir, save_dir=save_dir)



# # move xmls
# jpgs_dir = r"D:\ddesktop\monitoring\datadata\helmet\helmetDataset\headDataset\jpgs"
# xmls_dir = r"D:\ddesktop\monitoring\datadata\helmet\helmetDataset\heads_xmls"
# save_dir = r"D:\ddesktop\monitoring\framesdata\dataset\xmls"
# moveXml_byjpg(jpg_dir=jpgs_dir, xml_dir=xmls_dir, save_dir=save_dir)


jpgs_dir = r"D:\ddesktop\monitoring\framesdata\dataset\jpgs"
xmls_dir = r"D:\ddesktop\monitoring\framesdata\dataset\xmls"

jpgname_set = {f.split('.')[0] for f in os.listdir(jpgs_dir) if f.endswith('.jpg')}
xmlname_set = {f.split('.')[0] for f in os.listdir(xmls_dir) if f.endswith('.xml')}

print(len(jpgname_set))
print(len(xmlname_set))
print(jpgname_set - xmlname_set)
