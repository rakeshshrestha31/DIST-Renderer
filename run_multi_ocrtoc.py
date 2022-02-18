import numpy as np
import os, sys
import cv2
import torch
from tqdm import tqdm
import easydict
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.dataset import LoaderMultiOCRTOC
from core.utils.render_utils import *
from core.utils.decoder_utils import load_decoder
from core.visualize.visualizer import print_loss_pack_color, Visualizer
from core.visualize.vis_utils import *
from core.evaluation import *
from core.sdfrenderer import SDFRenderer_warp
from core.inv_optimizer import optimize_multi_view

LR = 1e-2
THRESHOLD = 5e-5


class_type = ['planes', 'chairs', 'cars', 'sofas']

def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Color training pipeline."
    )
    arg_parser.add_argument('-g', '--gpu', default='0', help='gpu id.')
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="2000",
        help='The checkpoint weights to use. This can be a number indicated an epoch or "latest" '
        + "for the latest weights (this is the default)",
    )
    arg_parser.add_argument('--test_step', '-t', type=int, default=5, help='test step.')
    arg_parser.add_argument('--visualize', action='store_true', help='visualization flag.')
    arg_parser.add_argument('--data_path', default='/3d-future/ocrtoc-rendered', help='path to PMO dataset.')
    arg_parser.add_argument('--obj_name', default='sofas', help='deepsdf class model for experiments. (support "planes", "chairs", "cars"')
    arg_parser.add_argument('--scale', type=float, default=0.0933, help='scale the size of input image, 224x224 -> 112x112.')
    arg_parser.add_argument('--focal', type=float, default=None, help='resize the image and change focal length, try 2')
    arg_parser.add_argument('--full', action='store_true', help='run over all PMO data, otherwise run demo')

    args = arg_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ################################
    num_sample_points = 10000
    num_views_per_round = 8
    sep_dist = 1
    refine_sim = True
    ################################

    # load data
    class_id = args.obj_name
    exp_dir = os.path.join('deepsdf/experiments/', args.obj_name)

    upper_loader = LoaderMultiOCRTOC(args.data_path, class_id, scale=args.scale, num_points=num_sample_points, focal=args.focal)
    if args.full:
        total_num_instance = 50 # consider 50 instances in total
        out_dir = os.path.join('vis/multiview_syn/',  args.obj_name)
    else:
        total_num_instance = len(upper_loader) # demo data
        out_dir = os.path.join('vis/demo_multiview_syn/',  args.obj_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cf_dist1_total = 0.0
    cf_dist2_total = 0.0

    for instance_num in range(total_num_instance):
        vis_dir = os.path.join(out_dir, '{}'.format(instance_num))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        instance_name, imgs, masks, cameras, points_gt = upper_loader[instance_num]

        # visualize_3d(points_gt, cameras, args.obj_name)
        # exit(0)

        # RANDOMLY initialize shape code
        latent_size = 256
        std_ = 0.1
        shape_code = torch.ones(1, latent_size).normal_(mean=0, std=std_)
        shape_code = shape_code.float().cuda()
        shape_code.requires_grad = True

        decoder = load_decoder(exp_dir, args.checkpoint)
        decoder = decoder.module.cuda()
        optimizer_latent = torch.optim.Adam([shape_code], lr=LR)

        img_h, img_w = imgs[0].shape[0], imgs[0].shape[1]
        img_hw = (img_h, img_w)
        print('Image size: {0}.'. format(img_hw))
        sdf_renderer = SDFRenderer_warp(
            decoder, cameras[0].intrinsic, march_step=200, buffer_size=1, threshold=THRESHOLD,
            transform_matrix=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        )
        evaluator = Evaluator(decoder)
        visualizer = Visualizer(img_hw)

        # with torch.no_grad():
        #     visualize_renderer(
        #         sdf_renderer, evaluator, shape_code, imgs, cameras,
        #         points_gt, args.obj_name
        #     )
        #     exit(0)

        weight_list = {}
        weight_list['color'] = 5.0
        weight_list['l2reg'] = 1.0

        shape_code, optimizer_latent = optimize_multi_view(sdf_renderer, evaluator, shape_code, optimizer_latent, imgs, cameras, weight_list, num_views_per_round=num_views_per_round, num_iters=50, sep_dist=sep_dist, num_sample_points=num_sample_points, visualizer=visualizer, points_gt=points_gt, vis_dir=vis_dir, vis_flag=args.visualize, full_flag=args.full)

        if args.full:
            # final evaluation
            points_tmp = evaluator.latent_vec_to_points(shape_code, num_points=num_sample_points, silent=True)
            dist1, dist2 = evaluator.compute_chamfer_distance(points_gt, points_tmp, separate=True)

            cf_dist1_total += dist1 * 1000
            cf_dist2_total += dist2 * 1000

    if args.full:
        print('Final Average Chamfer Loss: ', cf_dist1_total / total_num_instance, cf_dist2_total / total_num_instance)
    print('Finished. check results {}'.format(out_dir))


def visualize_3d(points_gt, cameras, obj_name):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_gt)
    axes = [
        o3d.geometry.TriangleMesh
            .create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            .transform(
                np.linalg.inv(np.concatenate((camera.extrinsic, [[0, 0, 0, 1]]), axis=0))
            )
        for camera in cameras
    ]
    mesh = o3d.io.read_triangle_mesh(f'vis/demo_multiview_syn/{obj_name}/0/mesh_initial.ply')

    o3d.visualization.draw_geometries([pcd, mesh] + axes[:1])


def depth_to_point_cloud(depth, camera):
    T_world_camUEN = np.linalg.inv(np.concatenate(
        (camera.extrinsic, [[0, 0, 0, 1]]), axis=0
    ))

    # DIST has weird pose convention (Up-East-North). Convert to East-Down-North
    R_EDN_UEN = np.asarray([
        0, 1, 0, -1, 0, 0, 0, 0, 1
    ]).reshape((3, 3))

    T_world_camEDN = T_world_camUEN.copy()
    T_world_camEDN[:3, :3] = T_world_camUEN[:3, :3] @ R_EDN_UEN.T

    h, w = depth.shape

    depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image((depth.detach().cpu().numpy()).astype(np.float32)),
        o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h,
            fx=camera.intrinsic[0, 0], fy=camera.intrinsic[1, 1],
            cx=camera.intrinsic[0, 2], cy=camera.intrinsic[1, 2],
        ),
        depth_scale=1.0
    ).transform(T_world_camEDN)

    return depth_pcd


def visualize_renderer(
        sdf_renderer, evaluator, shape_code, imgs, cameras, points_gt, obj_name,
        sim3=None, sim3_init=None
):
    viz_imgs = []
    depth_pcds = []
    h, w = imgs[0].shape[:2]

    from core.utils.train_utils import params_to_mtrx
    if sim3 is not None:
        sim_mtrx = params_to_mtrx(sim3).clone()
        sim_mtrx[:, 3] = torch.matmul(sim_mtrx[:3, :3].clone(), sim3_init[:, 3]) + sim_mtrx[:, 3].clone()
        sim_mtrx[:3, :3] = torch.matmul(sim_mtrx[:3, :3].clone(), sim3_init[:3, :3])
        sim3_scale = torch.norm(sim_mtrx[:3, :3].clone())/np.sqrt(3)
    else:
        sim_mtrx = None
        sim3_scale = None

    extrinsics = []
    for idx, (camera, img) in enumerate(zip(cameras, imgs)):
        R, T = camera.extrinsic[:,:3], camera.extrinsic[:,3]
        R, T = torch.from_numpy(R).float().cuda(), torch.from_numpy(T).float().cuda()

        if sim3 is not None:
            T = torch.matmul(R, sim_mtrx[:, 3]) + T
            R = torch.matmul(R, sim_mtrx[:3, :3])
            R = R / sim3_scale
            T = T / sim3_scale

        extrinsic = torch.from_numpy(camera.extrinsic).float().cuda()
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        extrinsics.append(extrinsic)

        depth, mask, _ = sdf_renderer.render_depth(
            shape_code, R, T, no_grad=True
        )
        depth[~mask] = 0
        depth = depth.reshape(h, w).contiguous()

        depth_colored = depth.unsqueeze(-1).expand(-1, -1, 3) \
                                .detach().cpu().numpy()
        rgb = img.detach().cpu().numpy()
        viz_img = np.concatenate((depth_colored, rgb), axis=1)
        viz_img = (viz_img * 255).astype(np.uint8)
        viz_imgs.append(viz_img)
        depth_pcds.append(depth_to_point_cloud(depth, camera))

    viz_imgs = np.concatenate(viz_imgs, axis=0)
    cv2.imwrite('/tmp/viz_imgs.png', viz_imgs)

    # gt_pcd = o3d.geometry.PointCloud()
    # gt_pcd.points = o3d.utility.Vector3dVector(points_gt)

    T_world_camsUEN = [
        np.linalg.inv(np.concatenate(
            (extrinsic.detach().cpu().numpy(), [[0, 0, 0, 1]]), axis=0
        ))
        for extrinsic in extrinsics
    ]
    axes_UEN = [
        o3d.geometry.TriangleMesh
            .create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            .transform(T_world_cam)
        for T_world_cam in T_world_camsUEN
    ]

    evaluator.latent_vec_to_points(
        shape_code, num_points=30000, fname='/tmp/mesh_initial.ply', silent=False
    )
    mesh = o3d.io.read_triangle_mesh('/tmp/mesh_initial.ply')

    o3d.visualization.draw_geometries([
        # *depth_pcds,
        mesh,
        *axes_UEN[0:1], # gt_pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    ])


if __name__ == '__main__':
    # seed = 123
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # import random
    # random.seed(seed)
    main()

