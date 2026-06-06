import random

import cv2
import os

import tempfile
import threading
from subprocess import call

import numpy as np
from scipy.io import wavfile
import pyrender

import librosa

from tqdm import tqdm

import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # Uncommnet this line while running remotely
import cv2
import pyrender
import trimesh
import tempfile
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None,
                       errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=1.0, xmag=0.5,
                       y=0.7, z=1, camera='o', r=None):
    camera_params = {'c': np.array([0, 0]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([5000, 5000])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    v, f = mesh
    v = cv2.Rodrigues(rot)[0].dot((v - t_center).T).T + t_center

    texture_rendering = tex_img is not None and hasattr(mesh, 'vt') and hasattr(mesh, 'ft')
    if texture_rendering:
        intensity = 0.5
        tex = pyrender.Texture(source=tex_img, source_channels='RGB')
        material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)

        # Workaround as pyrender requires number of vertices and uv coordinates to be the same
        temp_filename = '%s.obj' % next(tempfile._get_candidate_names())
        mesh.write_obj(temp_filename)
        tri_mesh = trimesh.load(temp_filename, process=False)
        try:
            os.remove(temp_filename)
        except:
            print('Failed deleting temporary file - %s' % temp_filename)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
    elif errors is not None:
        intensity = 0.5
        unit_factor = get_unit_factor('mm') / get_unit_factor(error_unit)
        errors = unit_factor * errors

        norm = mpl.colors.Normalize(vmin=min_dist_in_mm, vmax=max_dist_in_mm)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba_per_v = colormapper.to_rgba(errors)
        rgb_per_v = rgba_per_v[:, 0:3]
    elif v_colors is not None:
        intensity = 0.5
        rgb_per_v = v_colors
    else:
        intensity = 6.
        rgb_per_v = None

    color = np.array([0.3, 0.5, 0.55])

    if not texture_rendering:
        tri_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh,
                                                 smooth=True,
                                                 material=pyrender.MetallicRoughnessMaterial(
                                                     metallicFactor=0.05,
                                                     roughnessFactor=0.7,
                                                     alphaMode='OPAQUE',
                                                     baseColorFactor=(color[0], color[1], color[2], 1.0)
                                                 ))

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    if camera == 'o':
        ymag = xmag * z_offset
        camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    elif camera == 'i':
        camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                           fy=camera_params['f'][1],
                                           cx=camera_params['c'][0],
                                           cy=camera_params['c'][1],
                                           znear=frustum['near'],
                                           zfar=frustum['far'])
    elif camera == 'y':
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 2.0))

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0.7, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, y],  # 0.25
                            [0, 0, 1, z],  # 0.2
                            [0, 0, 0, 1]])


    angle = np.pi / 6.0
    # pos = camera_pose[:3,3]
    pos = np.array([0, 0.7, 2.0])
    if False:
        light_color = np.array([1., 1., 1.])
        light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 0.7, 2.0])
        scene.add(light, pose=light_pose.copy())
    else:
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=2)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [-1, 1, 2]
        scene.add(light, pose=light_pose)

        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                                    innerConeAngle=np.pi / 3, outerConeAngle=np.pi / 2)

        light_pose[:3, 3] = [-1, 2, 2]
        scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [1, 2, 2]
        scene.add(spot_l, pose=light_pose)

    # light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())
    #
    # light_pose[:3,3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())
    #
    # light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())
    #
    # light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    # pyrender.Viewer(scene)

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    # try:
    # r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=flags)
    # r.delete()
    # except:
    #     print('pyrender: Failed rendering frame')
    #     color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]



class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get_sen(i, num_video, i_frame, pos):
    if num_video == 1:
        sen = 'GT'
    elif num_video == 2:
        if i == 0:
            if pos == 1:
                sen = 'A'
            elif pos == 2:
                sen = 'B'
            else:
                sen = 'GT'
        else:
            if pos == 1:
                sen = 'B'
            elif pos == 2:
                sen = 'A'
            else:
                sen = 'result'
    elif num_video == 3:
        if i == 0:
            sen = 'sample1'
        elif i == 1:
            sen = 'interpolation'
        else:
            sen = 'sample2'
    elif num_video == 9 or num_video == 16:
        if i == 0:
            sen = 'frame '+str(i_frame)
        else:
            sen = 'sample' + str(i)
    elif num_video == 12:
        if i == 0:
            sen = 'sample1'
        elif i < 11:
            sen = 'interpolation' + str(i)
        else:
            sen = 'sample2'

    return sen


def add_image_text(img, text, color=(0,0,255), w=800, h=800):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 8, 2)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    textY = textsize[1] + 10
    # img = img.copy()
    # a = img * 255
    # img = a.transpose(1, 2, 0).astype(np.uint8).copy()
    # cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # w = int(text)

    # img = img.transpose(1, 2, 0)
    img = np.require(img, dtype='f4', requirements=['O', 'W'])
    img.flags.writeable = True
    img1 = img.copy()
    img1 = cv2.putText(img1, '%s' % (text), (100, 100), font, 4, color, 2, 1)
    img1 = cv2.rectangle(img1, (0, 0), (w, h), color, thickness=3, )

    # img1 = img1.transpose(2, 0, 1)

    return img1


class RenderTool():
    def __init__(self, out_path):
        path = "/mnt/data/cbh/SynTalker/datasets/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
        model_data = np.load(path, allow_pickle=True)
        data_struct = Struct(**model_data)
        self.f = data_struct.f
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def _render_sequences(self, cur_wav_file, v_list, j=-1, stand=False, face=False, whole_body=False, run_in_parallel=False, transcript=None, gt=False,index=0,numpy_audio=None):
        # import sys
        # if sys.platform == 'win32':
        if gt:
            suffix = 'gt'
        else:
            suffix = 'gen'
        suffix+=str(index)
        
        symbol = '/'
        # else:
        #     symbol = '\\'
        print("Render {} {} sequence.".format(cur_wav_file.split(symbol)[-2],cur_wav_file.split(symbol)[-1]))
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(cur_wav_file, v_list))
            thread.start()
            thread.join()
        else:
            # directory = os.path.join(self.out_path, cur_wav_file.split(symbol)[-2])
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # video_fname = os.path.join(directory, '%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2])
            directory = self.out_path
            if not os.path.exists(directory):
                os.makedirs(directory)
            if j == -1:
                video_fname = os.path.join(directory, '%s.mp4' % suffix)
            elif j == -2:
                video_fname = os.path.join(directory, cur_wav_file.split(symbol)[-3]+'--%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split(symbol)[-1])
            else:
                video_fname = os.path.join(directory, str(j)+'_%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split(symbol)[-1])
            self._render_sequences_helper(video_fname, cur_wav_file, v_list, stand, face, whole_body, transcript, numpy_audio)

    def _render_sequences_helper(self, video_fname, cur_wav_file, v_list, stand, face, whole_body, transcript, numpy_audio=None):
        num_frames = v_list[0].shape[0]

        # dataset is inverse
        for v in v_list:
            v = v.reshape(v.shape[0], -1, 3)
            v[:, :, 1] = -v[:, :, 1]
            v[:, :, 2] = -v[:, :, 2]
        viewport_height = 800
        z_offset = 1.0
        num_video = len(v_list)
        assert num_video in [1, 2, 3, 9, 12, 16, 18]
        if num_video == 1:
            width, height = 800, 800
        elif num_video == 2:
            width, height = 1600, 800
        elif num_video == 3:
            width, height = 2400, 800
        elif num_video == 9:
            width, height = 2400, 2400
        elif num_video == 12:
            width, height = 3200, 2400
        elif num_video == 16:
            width, height = 3200, 3200
        elif num_video == 18:
            width, height = 4800, 2400

        if whole_body:
            width, height = 800, 1440
            viewport_height = 1440
            z_offset = 1.8

        sr = 22000
        if numpy_audio is None:
            audio, sr = librosa.load(cur_wav_file, sr=16000)
        else :
            audio = numpy_audio.squeeze()
            sr = 16000
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        tmp_audio_file.close()
        wavfile.write(tmp_audio_file.name, sr, audio)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        tmp_video_file.close()
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 30, (width, height), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), True)

        center = np.mean(v_list[0][0], axis=0)

        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=viewport_height)

        # random exchange the position of our method and SG3D
        # pos = random.randint(1, 2)
        # video_fname = list(video_fname)
        # video_fname.insert(-4, str(pos))
        # video_fname = ''.join(video_fname)
        pos = 1

        for i_frame in tqdm(range(num_frames)):
            # pyrender.Viewer(scene)
            cur_img = []
            for i in range(len(v_list)):
                if face:
                    img = render_mesh_helper((v_list[i][i_frame], self.f), center,
                                             r=r, xmag=0.15, y=1, z=1.0, camera='o')
                else:
                    img = render_mesh_helper((v_list[i][i_frame], self.f), center, camera='o', r=r, y=0.7, z_offset=z_offset)
                # sen = get_sen(i, num_video, i_frame, pos)
                # if transcript is not None:
                #     sen = str(int(transcript[i_frame].item()))
                # else:
                #     sen = ' '
                # img = add_image_text(img, sen)
                cur_img.append(img)

            if num_video == 1:
                final_img = cur_img[0].astype(np.uint8)
            elif num_video == 2:
                final_img = np.hstack((cur_img[0], cur_img[1])).astype(np.uint8)
            elif num_video == 3:
                final_img = np.hstack((cur_img[0], cur_img[1], cur_img[2])).astype(np.uint8)
            elif num_video == 9:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[3], cur_img[4], cur_img[5])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[6], cur_img[7], cur_img[8])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            elif num_video == 12:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[4], cur_img[5], cur_img[6], cur_img[7])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            elif num_video == 16:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[4], cur_img[5], cur_img[6], cur_img[7])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                img_vert_3 = np.hstack((cur_img[12], cur_img[13], cur_img[14], cur_img[15])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2, img_vert_3)).astype(np.uint8)
            elif num_video == 18:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3], cur_img[4], cur_img[5])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[6], cur_img[7], cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[12], cur_img[13], cur_img[14], cur_img[15], cur_img[16], cur_img[17])).astype(
                    np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            # final_img = add_image_text(final_img, 'frame'+str(i_frame), w=width, h=height)
            writer.write(final_img)
        writer.release()

        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -acodec mp3 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
        print(cmd)
        # cmd = ('ffmpeg' + '-i {0} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {1}'.format(
        #     tmp_video_file.name, video_fname)).split()
        call(cmd)
        os.remove(tmp_audio_file.name)
        os.remove(tmp_video_file.name)


