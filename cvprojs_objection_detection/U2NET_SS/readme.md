# u2net   semantic segmentation

'file system
/raw_data/
    +--------/raw/  may raw jpgs, rgb pngs, jsons, ...
    +--------/datadata/   split --> dataset: train_data, test_data
/train_data/
    +--------/train_jpgs/
    +--------/train_pngs/
/test_data/
    +--------test_jpgs
    +--------test_pngs
/dataset/
    +--------data process utils
/train_and_eval/
    +-------- utils for train eval trian_one_epoch, criterion, 

/src/
    +-------- model.py

/model_data/
    +------- xx.tiff  weights

/logs/
    +------ checkpoints    --add train from checkpoint 

train.py

u2netSSengin.py   build a function: out_image = u2net_ss(jpg_input)
predict.py  process_input, use:u2net_ss, dir_predict or predict, save_and_show_result

validation.py
'

### 现在的数据集是处理错了的，  得到的彩色的pngs
#### 正确的方法1： labelme  .josn file --> pngs
#### 2:只有jpgs rgb-pngs，  rgb-pngs --> pngs
