import os
import pathlib
from collections import defaultdict
from importlib.resources import path

import click
import numpy as np
import pymeshlab


def add_vn(dir, filename):
  if filename[-3:] != 'obj':
    return
  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(os.path.join(dir, 'c_'+filename))
  ms.save_current_mesh(file_name=os.path.join(dir, 'cn_'+filename),
  save_face_color=False, save_wedge_texcoord=False,
  save_wedge_normal=False, save_polygonal=False)


def normalize_grabcadprint(mesh_path, target_mesh_path, ground_truth, scale=10):
  """
    mesh_path: path of .obj files (without v_normals)
    ground_truth: the true length of axis z, mm
    scale: the scale of ground_truth unit to grabcad print unit (1cm->1mm, x10)
  """
  vertices = []
  faces = []
  colors = []
  with open(mesh_path, 'r') as f:
    for line in f.readlines():
      content = line.split()
      if len(content) == 0:
        continue
      if content[0] == 'v':
        vertices.append([float(content[1]), float(content[2]), float(content[3])])
        if len(content) == 7:
          colors.append([float(content[4]), float(content[5]), float(content[6])])
        else:
          colors.append([])
      elif content[0] == 'f':
        faces.append([float(content[1].split('/')[0]), float(content[2].split('/')[0]), float(content[3].split('/')[0])])
  vertices_np = np.array(vertices)
  boundings = np.ptp(vertices_np, axis=0).tolist()
  max_bounding = max(boundings)
  # ground_truth 18cm
  # scale 100
  # max_bounding 17 (mm)
  target_scale = ground_truth * scale / max_bounding
  vertices_np = vertices_np * target_scale
  vertices = vertices_np.tolist()

  with open(target_mesh_path, 'w') as f:
    for i, v in enumerate(vertices):
      if len(colors[i]) == 0:
        f.write('v {} {} {}\n'.format(float(v[0]),float(v[1]),float(v[2])))
      else:
        f.write('v {} {} {} {} {} {}\n'.format(float(v[0]),float(v[1]),float(v[2]),float(colors[i][0]),float(colors[i][1]),float(colors[i][2])))
    for face in faces:
      f.write('f {} {} {}\n'.format(int(face[0]),int(face[1]),int(face[2])))

def read_mesh_with_vn(mesh_path):
  front_v_normals = []
  front_vertices = []
  front_faces = []
  front_colors = []
  with open(mesh_path, 'r') as f:
    for line in f.readlines():
      content = line.split()
      if len(content) == 0:
        continue
      if content[0] == 'vn':
        front_v_normals.append([float(content[1]), float(content[2]), float(content[3])])
      elif content[0] == 'v':
        front_vertices.append([float(content[1]), float(content[2]), float(content[3])])
        if len(content) == 7:
          front_colors.append([float(content[4]), float(content[5]), float(content[6])])
        else:
          front_colors.append([])
      elif content[0] == 'f':
        front_faces.append([float(content[1].split('/')[0]), float(content[2].split('/')[0]), float(content[3].split('/')[0])])
  return front_v_normals, front_vertices, front_faces, front_colors

def dfs(u, graph, visited, path):
  if u in visited:
    return
  visited.add(u)
  path.append(u)
  for neighbor in graph[u]:
    dfs(neighbor, graph, visited, path)

def free_boundary(faces):
  count = defaultdict(int)
  for face in faces:
    count[(face[0], face[1])] += 1
    count[(face[1], face[2])] += 1
    count[(face[2], face[0])] += 1

  boundary = set()
  for edge, cnt in count.items():
    if (edge[1], edge[0]) in count.keys():
      inv_cnt = count[(edge[1], edge[0])]
    else:
      inv_cnt = 0
    if cnt + inv_cnt >= 2:
      continue
    else:
      if (edge[1], edge[0]) not in boundary:
        boundary.add(edge)

  graph = defaultdict(list)
  for edge in boundary:
    graph[edge[0]].append(edge[1])
    graph[edge[1]].append(edge[0])
  
  if len(graph) == 0:
    return []
  
  visited = set()

  paths = []
  for k in graph.keys():
    if k not in visited:
      path = []
      dfs(k, graph, visited, path)
      paths.append(path)
  
  print(len(paths))

  return paths

def mesh_thicken(mesh_path, thickness=-2, target_obj='./double_obj.obj'):
  front_v_normals, front_vertices, \
    front_faces, front_colors = read_mesh_with_vn(mesh_path=mesh_path)
  
  assert len(front_v_normals) == len(front_vertices)

  front_vertex_num = len(front_vertices)

  front_v_normals_np = np.array(front_v_normals)
  front_v_normals_np = front_v_normals_np / np.linalg.norm(front_v_normals_np, axis=1, keepdims=True)
  print(front_v_normals_np)
  front_vertices_np = np.array(front_vertices)

  back_vertices_np = front_v_normals_np * thickness + front_vertices_np
  back_vertices = back_vertices_np.tolist()

  # all faces' vertices add an offset len(front_vertices)
  back_faces = [[v + front_vertex_num for v in face] for face in front_faces]

  # find front and back boundary
  front_boundaries = free_boundary(faces=front_faces)
  
  bridge_faces = []

  for front_boundary in front_boundaries:
    n = len(front_boundary)
    for i in range(0, n - 1):
      node = front_boundary[i]
      bridge_faces.append([node, front_boundary[i+1], node+front_vertex_num])
      bridge_faces.append([node+front_vertex_num, front_boundary[i+1], front_boundary[i+1]+front_vertex_num])
    # circle
    bridge_faces.append([front_boundary[-1], front_boundary[0], front_boundary[-1]+front_vertex_num])
    bridge_faces.append([front_boundary[-1]+front_vertex_num, front_boundary[0], front_boundary[0]+front_vertex_num])

  with open(target_obj, 'w') as f:
    for i, v in enumerate(front_vertices):
      if len(front_colors[i]) == 3:
        f.write('v {} {} {} {} {} {}\n'.format(v[0],v[1],v[2], front_colors[i][0], front_colors[i][1], front_colors[i][2]))
      else:
        f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        
    for i, v in enumerate(back_vertices):
      if len(front_colors[i - front_vertex_num]) == 3:
          f.write('v {} {} {} {} {} {}\n'.format(v[0],v[1],v[2], front_colors[i - front_vertex_num][0], front_colors[i - front_vertex_num][1], front_colors[i - front_vertex_num][2]))
      else:
        f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for fc in front_faces:
      f.write('f {} {} {}\n'.format(int(fc[0]), int(fc[1]), int(fc[2])))
    for fc in back_faces:
      f.write('f {} {} {}\n'.format(int(fc[2]), int(fc[1]), int(fc[0])))
    for fc in bridge_faces:
      f.write('f {} {} {}\n'.format(int(fc[2]), int(fc[1]), int(fc[0])))


@click.command()
@click.option('--model_path', default="./data", required=True, type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path), help='The path of your thickened model')
@click.option('--thickness', required=True, type=float, default=1.5, help='The thickness you want (mm)')
@click.option('--max_length', required=True, type=float, help='The max length of the bounding box (cm)')
def main(model_path, thickness, max_length):
  normalize_grabcadprint(model_path, model_path.parent / pathlib.Path('c_'+str(model_path.name)), max_length)
  add_vn(model_path.parent, model_path.name)
  mesh_thicken(model_path.parent / pathlib.Path('cn_'+str(model_path.name)), target_obj=model_path.parent / pathlib.Path('final_'+str(model_path.name)), thickness=-thickness)


if __name__=='__main__':
  main()
