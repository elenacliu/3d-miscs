import pathlib

import click
import numpy as np
import open3d as o3d


@click.command()
@click.option('--pc_path', type=click.Path(file_okay=True, path_type=pathlib.Path), 
            default='/data/jiegec/elenacliu/datasets/modelnet40/modelnet40_train.npz')
def main(pc_path):
    c = np.load(pc_path)
    xyz = c['tensor'][0, :, :]
    print("Printing numpy array used to make Open3D pointcloud ...")
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([4, 4, 4])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])

    # Convert Open3D.o3d.geometry.PointCloud to numpy array.
    xyz_converted = np.asarray(pcd.points)
    print("Printing numpy array made using Open3D pointcloud ...")
    print(xyz_converted)


if __name__ == "__main__":
    # Generate some n x 3 matrix using a variant of sync function.
    main()
