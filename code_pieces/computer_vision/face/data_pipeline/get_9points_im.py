from pathlib import Path
import shutil

ims_d = r"D:\ddesktop\robotss\train-label-1467\train_xiangdong1164\datas"
name_set = {x.stem for x in Path(ims_d).rglob('*.jpg')}
print(f"datas目录jpg数量: {len(name_set)}")

ims_8p = r"D:\ddesktop\robotss\train-label-1467\8-10-points\8_spots"
namegg_set = {x.stem for x in Path(ims_8p).rglob('*.jpg')}
print(f"8_spots目录jpg数量: {len(namegg_set)}")

# 检查交集
intersection = name_set & namegg_set
print(f"\n两个目录的交集数量: {len(intersection)}")
print(f"交集示例: {list(intersection)[:5]}")

# 如果交集为0，说明文件名不匹配
if len(intersection) == 0:
    print("\n❌ 问题：两个目录的文件名完全不匹配！")
    print("\ndatas目录示例文件名:")
    print(list(name_set)[:5])
    print("\n8_spots目录示例文件名:")
    print(list(namegg_set)[:5])
else:
    # 计算差集
    tgt_set = name_set - namegg_set
    print(f"\n过滤后应该复制的文件数量: {len(tgt_set)}")
    print(f"1036 - {len(intersection)} = {len(tgt_set)}")
    
    # 复制文件
    target_d = r"G:\temp\robot_head\facess\frames"
    Path(target_d).mkdir(parents=True, exist_ok=True)
    
    success = 0
    for name in tgt_set:
        jpgp_src = Path(ims_d) / f"{name}.jpg"
        pngp_src = Path(ims_d) / f"{name}.png"
        
        if jpgp_src.is_file() and pngp_src.is_file():
            jpgp_tgt = Path(target_d) / f"{name}.jpg"
            pngp_tgt = Path(target_d) / f"{name}.png"
            
            # shutil.copy2(jpgp_src, jpgp_tgt)
            # shutil.copy2(pngp_src, pngp_tgt)
            success += 1
    
    print(f"\n✅ 成功复制 {success} 对文件（jpg+png）")