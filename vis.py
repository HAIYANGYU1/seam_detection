
import numpy as np
import open3d as o3d
from pathlib import Path


obj_file_path = "./data/22-02-4002_1150TC-REV5_SSI_welding.obj"

def vis_obj(obj_file_path):
    file_path = Path(obj_file_path)

    if file_path.exists():
        print("exist")
    else:
        print("not exist")

    obj_mesh = o3d.io.read_triangle_mesh(obj_file_path)
    if not obj_mesh.has_vertex_normals():
        obj_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([obj_mesh])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(obj_mesh)
    vis.run()
    vis.destroy_window()

def vis_np_points(pnts_file_path,colors_file_path):
    points = np.load(pnts_file_path)
    colors = np.load(colors_file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    points_path = "pred_points.npy"
    vis_np_points(points_path,"pred_colors.npy")


