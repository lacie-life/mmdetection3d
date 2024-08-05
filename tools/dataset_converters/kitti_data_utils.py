# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image
from skimage import io

lidar2ego = np.asarray(
    [
        [0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
        [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
        [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

lidar2south1 = np.asarray(
    [
        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
    ],
    dtype=np.float32,
)

lidar2south2 = np.asarray(
    [
        [1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
        [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
        [0.73326062, 0.59708904, -0.32528854, -1.30114325],
    ],
    dtype=np.float32,
)

south1_intrinsic = np.asarray(
    [
        [1400.3096617691212, 0.0, 967.7899705163408],
        [0.0, 1403.041082755918, 581.7195041357244],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south12ego = np.asarray(
    [
        [-0.06377762, -0.91003007, 0.15246652, -10.409943],
        [-0.41296193, -0.10492031, -0.8399004, -16.2729],
        [0.8820865, -0.11257353, -0.45447016, -11.557314],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south12lidar = np.asarray(
    [
        [-0.10087585, -0.51122875, 0.88484734, 1.90816304],
        [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
        [0.01956882, -0.93122171, -0.45454375, 0.72290242],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south2_intrinsic = np.asarray(
    [
        [1029.2795655594014, 0.0, 982.0311857478633],
        [0.0, 1122.2781391971948, 1129.1480997238505],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south22ego = np.asarray(
    [
        [0.650906, -0.7435749, 0.15303044, 4.6059465],
        [-0.14764456, -0.32172203, -0.935252, -15.00049],
        [0.74466264, 0.5861663, -0.3191956, -9.351643],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south22lidar = np.asarray(
    [
        [0.49709212, -0.19863714, 0.64202357, -0.03734614],
        [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
        [0.01173726, -0.77546627, -0.70523436, 0.54322305],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]


def get_image_index_str(img_idx, use_prefix_id=False):
    type_lidar = img_idx[-5:]
    if use_prefix_id:
        img_idx = type_lidar + "/" + img_idx
        return str(img_idx)
    else:
        img_idx = type_lidar + "/" + img_idx
        return str(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('training') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2',
                   file_tail='.jpg',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_plane_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='planes',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_timestamp_path(idx,
                       prefix,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         with_plane=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        image_info['image_path'] = get_image_path(idx, path, training,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            
            # Check prefix for calib
            # print('idx:', idx)
            # print('path:', path)
            # print('calib_path:', calib_path)

            camera_type = idx[0:6]
            lidar_type = idx[-5:]
            # print('camera_type:', camera_type)
            # print('lidar_type:', lidar_type)

            R0_tmp = np.eye(4)
            Tr_v2cam_tmp = None
            cam_intrinsic = None

            if camera_type == 'south1':
                # cam_intrinsic = south1_intrinsic

                if lidar_type == 'south':
                    cam_intrinsic = np.array([
                        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
                    ])
                    Tr_v2cam_tmp = np.array(
                        [[-0.0931837, -0.995484, 0.018077, -13.8309],
                         [-0.481033, 0.029117, -0.876219, 1.96067],
                         [0.871736, -0.0903453, -0.481574, -2.58546],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )
                elif lidar_type == 'north':
                    cam_intrinsic = np.array(
                        [[2.90064402e+02, -1.52265886e+03, -4.17082935e+02, -3.98031250e+02],
                         [-8.89628396e+01, 6.86258619e+00, -1.45178013e+03, 4.54220718e+02],
                         [8.18463845e-01, -3.28184922e-01, -4.71605571e-01, -1.82577035e-01]]
                    )
                    Tr_v2cam_tmp = np.array(
                        [[-0.374855, -0.926815, 0.0222604, -0.284537],
                         [-0.465575, 0.167432, -0.869026, 0.683219],
                         [0.8017, -0.336123, -0.494264, -0.837352],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )
                else:
                    raise ValueError('Invalid lidar type')
            elif camera_type == 'south2':
                # cam_intrinsic = south2_intrinsic

                if lidar_type == 'south':
                    cam_intrinsic = np.array(
                        [[1318.95273325, -859.15213894, -289.13390611, 11272.03223502],
                         [90.01799314, -2.9727517, -1445.63809767, 585.78988153],
                         [0.876766, 0.344395, -0.335669, -7.26891]])
                    Tr_v2cam_tmp = np.array(
                        [[0.641509, -0.766975, 0.0146997, 1.99131],
                         [-0.258939, -0.234538, -0.936986, 1.21464],
                         [0.722092, 0.597278, -0.349058, -1.50021],
                         [0.0, 0.0, 0.0, 1.0]]
                    )
                elif lidar_type == 'north':
                    cam_intrinsic = np.array(
                        [[1.31895273e+03, -8.59152139e+02, -2.89133906e+02, 1.12720322e+04],
                         [9.00179931e+01, -2.97275170e+00, -1.44563810e+03, 5.85789882e+02],
                         [8.76766000e-01, 3.44395000e-01, -3.35669000e-01, -7.26891000e+00]]
                    )
                    Tr_v2cam_tmp = np.array(
                        [[0.37383, -0.927155, 0.0251845, 14.2181],
                         [-0.302544, -0.147564, -0.941643, 3.50648],
                         [0.876766, 0.344395, -0.335669, -7.26891],
                         [0.0, 0.0, 0.0, 1.0]
                         ]
                    )
                else:
                    raise ValueError('Invalid lidar type')

            
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = cam_intrinsic
            calib_info['P3'] = P3
            calib_info['R0_rect'] = R0_tmp
            calib_info['Tr_velo_to_cam'] = Tr_v2cam_tmp
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        if with_plane:
            plane_path = get_plane_path(idx, path, training, relative_path)
            if relative_path:
                plane_path = str(root_path / plane_path)
            lines = mmengine.list_from_file(plane_path)
            info['plane'] = np.array([float(i) for i in lines[3].split()])

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


class WaymoInfoGatherer:
    """
    Parallel version of waymo dataset information gathering.
    Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """

    def __init__(self,
                 path,
                 training=True,
                 label_info=True,
                 velodyne=False,
                 calib=False,
                 pose=False,
                 extend_matrix=True,
                 num_worker=8,
                 relative_path=True,
                 with_imageshape=True,
                 max_sweeps=5) -> None:
        self.path = path
        self.training = training
        self.label_info = label_info
        self.velodyne = velodyne
        self.calib = calib
        self.pose = pose
        self.extend_matrix = extend_matrix
        self.num_worker = num_worker
        self.relative_path = relative_path
        self.with_imageshape = with_imageshape
        self.max_sweeps = max_sweeps

    def gather_single(self, idx):
        root_path = Path(self.path)
        info = {}
        pc_info = {'num_features': 6}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if self.velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                use_prefix_id=True)
        with open(
                get_timestamp_path(
                    idx,
                    self.path,
                    self.training,
                    relative_path=False,
                    use_prefix_id=True)) as f:
            info['timestamp'] = np.int64(f.read())
        image_info['image_path'] = get_image_path(
            idx,
            self.path,
            self.training,
            self.relative_path,
            info_type='image_0',
            file_tail='.jpg',
            use_prefix_id=True)
        if self.with_imageshape:
            img_path = image_info['image_path']
            if self.relative_path:
                img_path = str(root_path / img_path)
            # io using PIL is significantly faster than skimage
            w, h = Image.open(img_path).size
            image_info['image_shape'] = np.array((h, w), dtype=np.int32)
        if self.label_info:
            label_path = get_label_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                info_type='label_all',
                use_prefix_id=True)
            cam_sync_label_path = get_label_path(
                idx,
                self.path,
                self.training,
                self.relative_path,
                info_type='cam_sync_label_all',
                use_prefix_id=True)
            if self.relative_path:
                label_path = str(root_path / label_path)
                cam_sync_label_path = str(root_path / cam_sync_label_path)
            annotations = get_label_anno(label_path)
            cam_sync_annotations = get_label_anno(cam_sync_label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if self.calib:
            calib_path = get_calib_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if self.extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if self.extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            # TODO: naming Tr_velo_to_cam or Tr_velo_to_cam0
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam1 = np.array([
                float(info) for info in lines[7].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam2 = np.array([
                float(info) for info in lines[8].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam3 = np.array([
                float(info) for info in lines[9].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_velo_to_cam4 = np.array([
                float(info) for info in lines[10].split(' ')[1:13]
            ]).reshape([3, 4])
            if self.extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_velo_to_cam1 = _extend_matrix(Tr_velo_to_cam1)
                Tr_velo_to_cam2 = _extend_matrix(Tr_velo_to_cam2)
                Tr_velo_to_cam3 = _extend_matrix(Tr_velo_to_cam3)
                Tr_velo_to_cam4 = _extend_matrix(Tr_velo_to_cam4)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_velo_to_cam1'] = Tr_velo_to_cam1
            calib_info['Tr_velo_to_cam2'] = Tr_velo_to_cam2
            calib_info['Tr_velo_to_cam3'] = Tr_velo_to_cam3
            calib_info['Tr_velo_to_cam4'] = Tr_velo_to_cam4
            info['calib'] = calib_info

        if self.pose:
            pose_path = get_pose_path(
                idx,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)
            info['cam_sync_annos'] = cam_sync_annotations
            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4
            info['cam_sync_annos']['camera_id'] = info['cam_sync_annos'].pop(
                'score')

        sweeps = []
        prev_idx = idx
        while len(sweeps) < self.max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_velodyne_path(
                prev_idx,
                self.path,
                self.training,
                self.relative_path,
                exist_check=False,
                use_prefix_id=True)
            if_prev_exists = osp.exists(
                Path(self.path) / prev_info['velodyne_path'])
            if if_prev_exists:
                with open(
                        get_timestamp_path(
                            prev_idx,
                            self.path,
                            self.training,
                            relative_path=False,
                            use_prefix_id=True)) as f:
                    prev_info['timestamp'] = np.int64(f.read())
                prev_info['image_path'] = get_image_path(
                    prev_idx,
                    self.path,
                    self.training,
                    self.relative_path,
                    info_type='image_0',
                    file_tail='.jpg',
                    use_prefix_id=True)
                prev_pose_path = get_pose_path(
                    prev_idx,
                    self.path,
                    self.training,
                    relative_path=False,
                    use_prefix_id=True)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    def gather(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = list(range(image_ids))
        image_infos = mmengine.track_parallel_progress(self.gather_single,
                                                       image_ids,
                                                       self.num_worker)
        return list(image_infos)


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
