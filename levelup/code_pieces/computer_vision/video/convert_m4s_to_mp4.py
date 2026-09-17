
# .m4s 视频文件  音频文件  合并
# ## ffmpeg -i video.m4s -i voice.m4s -c:v copy -c:a copy output.mp4
def merge_video_audio(video_file, audio_file, output_file):
    command = ['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'copy', output_file]
    subprocess.run(command, check=True)

# video_file = 'video.m4s'
# audio_file = 'voice.m4s'
# output_file = 'output.mp4'

# merge_video_audio(video_file, audio_file, output_file)






# ffmpeg -i input.m4s -c copy output.mp4

import subprocess
def convert_m4s_to_mp4(input_file, output_file):
    command = ['ffmpeg', '-i', input_file, '-c', 'copy', output_file]
    subprocess.run(command, check=True)
# input_file = 'path_to_your_input_file.m4s'
# output_file = 'path_to_your_output_file.mp4'

# convert_m4s_to_mp4(input_file, output_file)



# cut   a-b
def cut_video(input_file, start_time, end_time, output_file):
    command = ['ffmpeg', '-i', input_file, '-ss', start_time, '-to', end_time, '-c', 'copy', output_file]
    subprocess.run(command, check=True)
def concat_videos(file_list, output_file):
    with open('filelist.txt', 'w') as file:
        for f in file_list:
            file.write(f"file '{f}'\n")
    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt', '-c', 'copy', output_file]
    subprocess.run(command, check=True)

# # 裁剪视频
# cut_video('output.mp4', '00:00:00', '00:01:58', 'part1.mp4')
# cut_video('output.mp4', '00:02:19', None, 'part2.mp4')  # 如果end_time为None，表示一直到视频结束

# # 合并视频
# concat_videos(['part1.mp4', 'part2.mp4'], 'output_final.mp4')






