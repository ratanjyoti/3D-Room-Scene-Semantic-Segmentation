# from main_pipeline import RoomSegmentationPipeline

# # Change these to match your room
# AREA = "Area_1"
# ROOM = "office_1"

# pipeline = RoomSegmentationPipeline(output_dir=f"output_{AREA}_{ROOM}")

# pipeline.run_complete_pipeline(
#     anno_path=f"/media/ratan/New Volume/projects_vs code/s3dis/dataset/raw1/Stanford3dDataset_v1.2_Aligned_Version/{AREA}/{ROOM}/Annotations",
#     room_name=f"{AREA}_{ROOM}",
#     voxel_size=0.02,
#     eps=0.05,
#     show_plots=True  # Set to True to see plots
# )



from main_pipeline import RoomSegmentationPipeline

AREA = input("Enter area (e.g., Area_1): ")
ROOM = input("Enter room (e.g., office_1): ")

pipeline = RoomSegmentationPipeline(output_dir=f"output_{AREA}_{ROOM}")

pipeline.run_complete_pipeline(
    anno_path=r"E:\3D-semantic\3D-Room-Scene-Semantic-Segmentation\dataset\Stanford3dDataset_v1.2_Aligned_Version/{AREA}/{ROOM}/Annotations",
    room_name=f"{AREA}_{ROOM}",
    voxel_size=0.02,
    eps=0.05,
    show_plots=True  
)
