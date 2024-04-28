import numpy as np
import open3d as o3d
import torch

from openpoints.utils import EasyConfig,cal_model_parm_nums,load_checkpoint
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys


NUMBER_OF_SAMPLING_POINTS = 5_000

def obj2pcd(obj_file_path):
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=NUMBER_OF_SAMPLING_POINTS) 
    
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

   
    points_and_noprmal_concat = np.hstack((points, normals))
    # print(res.shape)
    num_samples = 2048
    sampled_idx = np.random.choice(points_and_noprmal_concat.shape[0], num_samples, replace=False)
    sampled_result = points_and_noprmal_concat[sampled_idx,:]
    
    return sampled_result

CONFIG_FILE_PATH = "cfgs/shapenetpart/pointnext-s.yaml"
PRETRAINED_PATH = "log/shapenetpart/shapenetpart-train-pointnext-s-ngpus1-seed6365-20240428-142658-MP6QqSnQ4R6VhqR8hNGZyo/checkpoint/shapenetpart-train-pointnext-s-ngpus1-seed6365-20240428-142658-MP6QqSnQ4R6VhqR8hNGZyo_ckpt_best.pth"
from collections import defaultdict, Counter

gravity_dim = 2


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]

def welding_seam_detect(obj_file_path):

    # 加载配置文件
    cfg = EasyConfig()
    cfg.load(CONFIG_FILE_PATH, recursive=True)
    model = build_model_from_cfg(cfg.model).cuda()
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    model.eval()

    point_set = obj2pcd(obj_file_path).astype(np.float32)

    # point_set = np.load("11644642_met_tl_asm.npy").astype(np.float32)

    point_set = point_set[np.newaxis, :, :]
    # point_set = point_set.reshape()
    # print(point_set.shape)


    # x = point_set[0:6]
    x = torch.from_numpy(point_set[:,:,3:]).cuda()
    pos = torch.from_numpy(point_set[:,:,0:3]).cuda()
    cls = torch.tensor([[1]]).cuda()
    
    data = {
        'pos':pos,
        'x':x,
        'cls':cls
    }

    # [x,y,z]
    height = data['pos'][:,:, gravity_dim:gravity_dim + 1]
    data['heights'] = height - torch.min(height)

    keys = ["pos","x","heights"]
    # res = get_features_by_keys(data, "pos,x,heights")

    res = torch.cat([data[key] for key in keys], -1)
    transpose_res = res.transpose(1,2).contiguous()
    data['x'] = transpose_res

    pred = model(data)
    preds = pred.max(dim=1)[1]
    preds_np = preds.detach().cpu().numpy()

    
    # [2,2,1] []


    
    pred = preds_np.reshape(2048,1)
    
    is_fine_pred = True 
    if is_fine_pred:
        unique_values, counts = np.unique(pred, return_counts=True)
        if (len(counts)) == 1:
            pred = np.full_like(pred,5.0)
        else:
            seam_value = unique_values[np.argmin(counts)]

            max_freq_value = unique_values[counts == counts.max()]
            pred = np.where(np.isin(pred, max_freq_value), 4, 5)
            
        unique_values, counts = np.unique(pred, return_counts=True)



    pos = data['pos'].detach().cpu().numpy().reshape(2048, 3)
    concatenated = np.concatenate((pred, pos), axis=1)
    return  concatenated

    # cls2parts = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23], [24, 25, 26, 27], [28, 29], [30, 31, 32, 33, 34, 35], [36, 37], [38, 39, 40], [41, 42, 43], [44, 45, 46], [47, 48, 49]]

    # part_seg_refinement(preds, data['pos'], data['cls'],cls2parts , cfg.get('refine_n', 10))




if __name__ == "__main__":
    obj_file_path = "11644642_met_tl_asm.obj"
    pred = welding_seam_detect(obj_file_path=obj_file_path)
    print(pred)
    np.savetxt("11644642_met_tl_asm.csv", pred, delimiter=',', fmt='%.6f')


    points = np.array(pred[:,1:])
    color_map = {
        4.0:[0.1,0.1,1.0],
        5.0:[0.1,1.0,0.1],
    }
    colors = np.array([color_map[value] for value in pred[:,0].flatten()])
    # print(points)
    np.save("pred_points.npy",points)
    np.save("pred_colors.npy",colors)

    # exit(0)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    # print(pred[0])
