from video_process import VideoProcess as VP
vp = VP("/home/ansel/works/datasets/DJI_20250725182253_0003_V.mp4")
# vp.add_gnss_path("/home/ansel/works/datasets/DJI_20250725182253_0003_V.SRT")
# vp.extract_frames(2)
exif_dict = vp.read_exif_from_image("/home/ansel/works/vggt-slam/temp_frames/frame_000001.jpg")
print(exif_dict["latitude"])