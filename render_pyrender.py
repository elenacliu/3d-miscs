import matplotlib

matplotlib.use('Agg')
import os
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def make_hyparam_str(intensity, color, ambient_light, bg_color):
    return 'intensity_{intensity}_color_{color}_ambientlight_{ambient_light}_bgcolor_{bg_color}'.format(
        bg_color=bg_color, ambient_light=ambient_light, color=color, intensity=intensity
    )


def render_one_obj(obj_path, pose_path, save_directory, bg_color=[0,0,0], intensity=1e3, color=[1,1,1], ambient_light=[0.02, 0.02, 0.02, 1.0]):
    if pose_path.suffix != '.npz':
        raise NotImplementedError('You should save all the pose matrices (4x4) in one npz file!')
    
    # generate mesh
    object = trimesh.load(obj_path, force='mesh')
    # compose scene

   
    poses = np.load(pose_path)
    for key in poses.files:
        # switch to "osmesa" or "egl" before loading pyrender
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        mesh = pyrender.Mesh.from_trimesh(object, smooth=False)
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)

        light = pyrender.DirectionalLight(color=color, intensity=intensity)
        scene = pyrender.Scene(ambient_light=ambient_light, bg_color=bg_color)

        scene.add(mesh, pose=np.eye(4))
        scene.add(light, pose=np.eye(4))
        scene.add(camera, pose=poses[key])

        # render scene
        r = pyrender.OffscreenRenderer(1024, 1024)
        Color, _ = r.render(scene)

        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(Color)
        print(key)
        plt.savefig(save_directory / pathlib.Path(key+'.png'), bbox_inches='tight', pad_inches=-0.5)
        plt.close()

        r.delete()
        scene.clear()

@click.command()
@click.option('--person_dir', type=click.Path(file_okay=False, path_type=pathlib.Path), help='The data dir of the victim')
@click.option('--intensity', type=float, default=1e3)
@click.option('--bg_color', type=click.Tuple([int, int, int]), default=(0,0,0))
@click.option('--ambient_light', type=list, default=[0.02, 0.02, 0.02, 1.0])
@click.option('--color', type=click.Tuple([int, int, int]), default=(1, 1, 1))
def render_all_mesh(person_dir, intensity, bg_color, ambient_light, color):
    bg_color = [*bg_color]
    color = [*color]

    datafiles = list(person_dir.glob('*.npz'))
    if len(datafiles):
        datafile = datafiles[0]
    else:
        raise NotImplementedError('Only support pose matrix in npz format files')

    for file in person_dir.iterdir():
        if file.is_dir() and (file / pathlib.Path('mesh')).exists():
            main(file / pathlib.Path('mesh'), datafile, intensity, bg_color, ambient_light, color)
    

# @click.command()
# @click.option('--mesh_dir', type=click.Path(file_okay=False, path_type=pathlib.Path), help='The path of directory that contains meshes')
# @click.option('--pose_path', type=click.Path(file_okay=True, path_type=pathlib.Path), help='The path of the pose npz file')
# @click.option('--intensity', type=float, default=1e3)
# @click.option('--bg_color', type=click.Tuple([int, int, int]), default=(0,0,0))
# @click.option('--ambient_light', type=list, default=[0.02, 0.02, 0.02, 1.0])
# @click.option('--color', type=click.Tuple([int, int, int]), default=(1, 1, 1))
def main(
    mesh_dir,
    pose_path,
    intensity,
    bg_color,
    ambient_light,
    color
):
    save_prefix = mesh_dir.parent / pathlib.Path('pyrender') / pathlib.Path(make_hyparam_str(intensity, color, ambient_light, bg_color))
    save_prefix.mkdir(parents=True, exist_ok=True)

    for file in mesh_dir.iterdir():
        if file.is_file() and file.suffix == '.obj':
            save_dir = save_prefix / file.stem
            print(save_dir)
            save_dir.mkdir(exist_ok=True)
            render_one_obj(file, pose_path, save_dir, intensity=intensity, color=color, ambient_light=ambient_light, bg_color=bg_color)

if __name__=='__main__':
    render_all_mesh()
