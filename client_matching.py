import xml.etree.ElementTree as ET
from pointpn.save_pn_feature import save_feature
from pointpn.cossim import pointpn
from pointnet2.main import pointnet2
from pointnext.main import pointnext
from ICP_RMSE import ICP
from poseE.main import poseestimation
import os.path
from tools import get_ground_truth,get_weld_info,WeldScene,image_save
from evaluation import mean_metric
import open3d as o3d
import numpy as np
import time
import shutil
# from model_splitter import split_models
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)

from welding_seam_detect import welding_seam_detect
from welding_seam_train import welding_seam_train,welding_seam_train_prepare,obj2txt,update_train_test_json


def matching(data_folder,xml_file,model,dienst_number,save_image=False,auto_del=False):
    if dienst_number==62:

        training_dir=os.path.join(ROOT,data_folder,'training')
        xml_list=os.listdir(training_dir)
        for file in xml_list:
            if not file.endswith('.xml'):
                continue
            xml_path=os.path.join(training_dir,file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            SNahts = root.findall("SNaht")
            Baugruppe = root.attrib['Baugruppe']
            wz_path = os.path.join(training_dir, Baugruppe)
            weld_infos = get_weld_info(xml_path)
            weld_infos = np.vstack(weld_infos)
            os.makedirs(wz_path, exist_ok=True)

            if not os.path.exists(os.path.join(training_dir,Baugruppe+'.pcd')):
                split(training_dir,Baugruppe)
                convert(training_dir,40,Baugruppe)
                print('creating pointcloud')
            ws = WeldScene(os.path.join(training_dir, Baugruppe + '.pcd'))
            for SNaht in SNahts:
                slice_name = SNaht.attrib['Name']
                # if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
                weld_info = weld_infos[weld_infos[:, 0] == slice_name][:, 3:].astype(float)
                if len(weld_info) == 0:
                    continue
                cxy, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(cxy)
                o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)
        if model == 'pointnet2':
            print('training pointnet2')
            os.system('python pointnet2/train_siamese_fortools.py --file_path data/training')
            print("pointnet2 training finished")
            return

        elif model == 'pointnext':
            print('training pointnext')
            os.system('python pointnext/classification/main.py --file_path data/training')
            print("pointnext training finished")
            return
    
    # pcd for training
    if dienst_number==63:
        welding_seam_train_prepare(data_folder)
        update_train_test_json()
        welding_seam_train()

    # obj for training
    if dienst_number==64:
        obj2txt(data_folder)
        update_train_test_json()
        welding_seam_train()

    # detect function
    if dienst_number==65:
        obj_file = xml_file.replace(".xml",".obj")
        obj_file=os.path.join(ROOT,data_folder,'train',obj_file)
        pred = welding_seam_detect(obj_file)
        print(f"pred={pred}")

    else:
        start_time=time.time()
        xml_path=os.path.join(ROOT,data_folder,xml_file)
        data_path = data_folder
        tree = ET.parse(xml_path)
        root = tree.getroot()
        SNahts = root.findall("SNaht")
        Baugruppe = root.attrib['Baugruppe']
        wz_path = os.path.join(data_path, Baugruppe)
        xml_output_path = os.path.join(ROOT, 'output')
        weld_infos=get_weld_info(xml_path)
        gt_id_map,gt_name_map=get_ground_truth(weld_infos)
        weld_infos=np.vstack(weld_infos)
        os.makedirs(xml_output_path, exist_ok=True)
        os.makedirs(wz_path,exist_ok=True)
        slice_name_list=[]
        if not os.path.exists(os.path.join(data_path,Baugruppe+'.pcd')):
            split(data_path,Baugruppe)
            convert(data_path,40,Baugruppe)

            inter_time=time.time()
            print('creating pointcloud time',inter_time-start_time)
        ws = WeldScene(os.path.join(data_path, Baugruppe + '.pcd'))
        for SNaht in SNahts:
            slice_name = SNaht.attrib['Name']
            # if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
            weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
            if len(weld_info)==0:
                continue
            slice_name_list.append(slice_name + '.pcd')
            cxy, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cxy)
            o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)

        retrieved_map={}
        methode_time=time.time()
        if model == 'icp':
            print('run icp')
            retrieved_map,retrieved_map_name,tree=ICP(SNahts,wz_path,tree,xml_path)

        elif model == 'pointpn':
            print('run pointpn')
            save_feature(wz_path,slice_name_list)
            retrieved_map=pointpn(SNahts,tree,xml_path)

        elif model == 'pointnet2':
            if dienst_number==63:
                print('run pointnet2')
                retrieved_map,retrieved_map_name,tree=pointnet2(wz_path,SNahts,tree,xml_path,slice_name_list)

        elif model == 'pointnext':
            if dienst_number==63:
                print('run pointnext')
                retrieved_map,retrieved_map_name,tree=pointnext(wz_path,SNahts,tree,xml_path,slice_name_list)
        # print('gt_map',gt_name_map)
        # print('retrieved_map_name',retrieved_map_name)
        #
        tree.write(os.path.join(xml_output_path, Baugruppe + '_similar.xml'))
        metric=mean_metric(gt_id_map,retrieved_map)
        print('metric',metric)
        if auto_del:
            shutil.rmtree(wz_path)

        if dienst_number==61:
            print('POSE ESTIMATION')
            tree=poseestimation(data_path,wz_path,xml_path,SNahts,tree,retrieved_map_name,vis=True)
            tree.write(os.path.join(xml_output_path, Baugruppe + '_predict.xml'))
        #
        # tree.write(os.path.join(xml_output_path,Baugruppe+'.xml'))
        if save_image:
            image_save(retrieved_map_name,wz_path)
        #
        print('gt_map',gt_id_map)
        print('retrieved_map',retrieved_map)
        #
        # metric=mean_metric(gt_id_map,retrieved_map)
        # print('metric',metric)
        # if auto_del:
        #     shutil.rmtree(wz_path)
        # end_time=time.time()
        # total_time=end_time-methoe_time
        # print(model,' running time= ',total_time)
        return

if __name__ == "__main__":

    data_folder=os.path.join(ROOT,'data')
    xml='Reisch_origin.xml'
    model='pointnext'
    dienst_number=62## 62 training_similarity;63 similarity; 61 pose estimation

    matching(data_folder, xml, model,dienst_number,save_image=False,auto_del=False)

