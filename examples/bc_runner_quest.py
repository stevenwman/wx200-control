import os
import queue
import time
import traceback
import multiprocessing as mp
import mujoco
import json
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
import mink
import socket

from src.sim.utils.bc_datalogger import BCDataLogger
from src.sim.utils.constants import QUEST3_FB2ID, PEG_INSERTION_KEY

start_ep = False
done_dc = False

class QuestListener:
    def __init__(self, T_a_mw, ip='0.0.0.0', port=6969, data_queue=None, selected_ids=None, scale=1.5):
        self.ip = ip
        self.port = port
        self.data_queue = data_queue
        self.socket = None
        self.running = False
        self.selected_ids = selected_ids
        self.scale = scale
        
        self.landmarks = np.zeros(len(self.selected_ids), dtype=object)
        self.calibration_frames = 0
        self.is_calibrated = False

        self.T_a_mw = T_a_mw
        self.T_a_w_mw = None
        
        self.vr2sc = R.from_euler('xyz',  [90,0,0], degrees=True)
    # np.array([pos['x'], pos['z'], pos['y']])
        
    def quest_to_w(self, pos, quat):
        pos = np.array([pos['x'], pos['z'], pos['y']])
        # rot = R.from_quat([quat['x'], quat['z'], quat['y'], quat['w']])
        rot = R.from_quat([0, 0, 0, 1])
        # .as_euler('xzy')
        # rot = R.from_euler('xyz', rot)
        return pos, rot

    def get_T_arm(self, pos, quat):
        pos, rot = self.quest_to_w(pos, quat)
        # return mink.SE3(np.concatenate((rot.as_quat(scalar_first=True), pos))), pos
        return mink.SE3.from_translation(pos), pos
    
    def get_T_f_a(self, pos, arm_pos):
        pos = (np.array([pos['x'], pos['z'], pos['y']]) - arm_pos) * self.scale
        return mink.SE3.from_translation(pos)
    
    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.ip, self.port))
        self.socket.settimeout(1)
        self.running = True
        print(f"Listening on {self.ip}:{self.port}")
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(6000)
                data = json.loads(data.decode('utf-8'))
                joints = data['joints']
                
                wrist = joints.pop(0)
                palm = joints.pop(0)
                # print(joints)
                T_a_w, arm_pos = self.get_T_arm(wrist['p'], wrist['r'])
                
                if not self.is_calibrated:
                    self.calibration_frames += 1
                    if self.calibration_frames > 10:
                        self.T_a_w_mw = self.T_a_mw @ T_a_w.inverse()
                        self.is_calibrated = True
                    continue
                
                T_a_mw = self.T_a_w_mw @ T_a_w
                
                idx = 0
                for joint in joints:
                    if joint['id'] not in self.selected_ids:
                        continue
                    
                    T_f_a = self.get_T_f_a(joint['p'], arm_pos)
                    self.landmarks[idx] = T_f_a
                    idx += 1
                    
                data_packet = {
                    'landmarks': self.landmarks.copy(),
                    'arm': T_a_mw.copy(),
                }
                
                try:
                    self.data_queue.put_nowait(data_packet)
                except queue.Full:
                    pass
                
            except socket.timeout:
                print('timeout')
                continue
            except Exception:
                print(traceback.format_exc())
                
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()

def receiver_process(T_a_mw, data_queue, selected_ids, scale):
    # T_a_mw, ip='0.0.0.0', port=6969, data_queue=None, selected_ids=None, scale=1.5
    receiver = QuestListener(T_a_mw, data_queue=data_queue, port=6969, selected_ids=selected_ids, scale=scale)
    try:
        receiver.start()
    except KeyboardInterrupt:
        receiver.stop()
        
def construct_ctrl_model(xml_str, fingers, bones) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_str)
    spec.add_key(name="home", qpos=PEG_INSERTION_KEY)

    body = spec.worldbody.add_body(name="ee_target", mocap=True)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.03,) * 3, rgba=(0.8, 0.2, 0.2, 0.7), contype=0, conaffinity=0)

    for finger in fingers:
        for bone in bones:
            body = spec.worldbody.add_body(name=f"{finger}_{bone}_target", mocap=True)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.01,) * 3, contype=0, conaffinity=0, rgba=(0.2, 0.8, 0.2, 0.7))
    return spec.compile()

def construct_ik_model(xml_str, fingers, bones) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml_str)
    spec.add_key(name="home", qpos=PEG_INSERTION_KEY[:-7])
    return spec.compile()
    
def set_mocap(model, data, T_f_mws, T_ee_mw, f_b_mocap_tgts):
    data.mocap_pos[model.body("ee_target").mocapid[0]] = T_ee_mw.translation()
    data.mocap_quat[model.body("ee_target").mocapid[0]] = T_ee_mw.rotation().wxyz
    
    for idx, f_b in enumerate(f_b_mocap_tgts):
        data.mocap_pos[model.body(f_b).mocapid[0]] = (T_f_mws[idx].translation())
        data.mocap_quat[model.body(f_b).mocapid[0]] = (T_f_mws[idx].rotation().wxyz)

def set_finger_tasks(tasks, T_f_ps, T_a_mw, selected_ids):
    T_f_mws = []
    for idx in range(len(selected_ids)):
        T_f_p = T_f_ps[idx]
        # print(T_a_mw, T_f_p)
        # T_f_mw = T_a_mw @ T_f_p
        T_f_mw = mink.SE3.from_translation(T_a_mw.translation() + T_f_p.translation())
        T_f_mws.append(T_f_mw)
        tasks[idx].set_target(T_f_mw)
    return T_f_mws

def key_callback(keycode):
    print("HAKUNA")
    global start_ep, end_ep, done_dc
    if chr(keycode) == ';':
        print("Starting Data Collection")
        start_ep = True
    elif chr(keycode) == '.':
        print("Ending Episode")
        start_ep = False
    elif chr(keycode) == ' ':
        print("Stopping Data Collection")
        done_dc = True
        
def bc_runner():
    # Define robot properties
    arm_ee_link = "attachment_site"
    # palm_body = "allegro_hand_right_palm_body"
    robot_hand = "allegro_hand_right"
    
    fingers = ['thumb', 'index', 'middle', 'ring']
    # bones = ["tip"]
    bones = ["prox", 'mid', 'dist', "tip"]
    th_bones = ["dist"]
    # bones = ["mcp", "prox", 'mid', "dist", "tip"]
    selected_ids = []
    f_b_bodies = []
    f_b_mocap_tgts = []
    for finger in fingers:
        if finger == 'thumb':
            for bone in th_bones:
                selected_ids.append(QUEST3_FB2ID[finger][bone])
                f_b_bodies.append(f"{robot_hand}_{finger}_{bone}_body")
                f_b_mocap_tgts.append(f"{finger}_{bone}_target")
        else:
            for bone in bones:
                selected_ids.append(QUEST3_FB2ID[finger][bone])
                f_b_bodies.append(f"{robot_hand}_{finger}_{bone}_body")
                f_b_mocap_tgts.append(f"{finger}_{bone}_target")

    # selected_ids = [QUEST3_FB2ID[f][b] for f in fingers for b in bones if b in QUEST3_FB2ID[f]]
    # f_b_bodies = [f"{robot_hand}_{f}_{b}_body" for f in fingers for b in bones  if b in QUEST3_FB2ID[f]]

    model_ik = construct_ik_model('./test_teleop/debug_robot_only.xml', fingers, bones)
    model_ctrl = construct_ctrl_model('./test_teleop/debug.xml', fingers, bones)
    data_ctrl = mujoco.MjData(model_ctrl)
    
    f_b_ids = [model_ctrl.body(body_name).id for body_name in f_b_bodies]
    peg_id = model_ctrl.body("peg").id
    hole_id = model_ctrl.body("hole").id

    configuration = mink.Configuration(model_ik)
    posture_task = mink.PostureTask(model=model_ik, cost=1e-2)
    end_effector_task = mink.FrameTask(frame_name=arm_ee_link, frame_type="site",
        position_cost=1.0, orientation_cost=0.0, lm_damping=1.0
    )

    finger_tasks = [mink.FrameTask(frame_name=f_b, frame_type="body",
        position_cost=20 if 'thumb' in f_b else 10.0, orientation_cost=0.0, lm_damping=5) for f_b in f_b_bodies
    ]

    tasks = [end_effector_task, posture_task, *finger_tasks]
    elbow_geoms = mink.get_body_geom_ids(model_ik, model_ik.body("wrist_1_link").id)
    elbow_geoms.extend(mink.get_body_geom_ids(model_ik, model_ik.body("wrist_2_link").id))
    collision_pairs = [(elbow_geoms, ['floor'])]
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(model=configuration.model, geom_pairs=collision_pairs,)
    ]
    solver = "daqp"
    model_ik = configuration.model
    data_ik = configuration.data

    arm_ee_id = model_ctrl.site(arm_ee_link).id
    quat_placeholder = np.zeros(4)
    home_pos = np.array([0.66, -0.196, 0.3132816])
    home_quat = R.from_euler('xyz', [90, 0, 143], degrees=True).as_quat(scalar_first=True)
    # T_a_mw = mink.SE3(np.array([*home_quat, *home_pos]))
    T_a_mw = mink.SE3(np.array([1, 0, 0, 0, *home_pos]))

    data_queue = mp.Queue(maxsize=5)
    receiver_proc = mp.Process(target=receiver_process, args=(T_a_mw, data_queue, selected_ids, 1.5))
    receiver_proc.start()

    # Reset both models to the home position
    mujoco.mj_resetDataKeyframe(model_ik, data_ik, model_ik.key("home").id)
    mujoco.mj_resetDataKeyframe(model_ctrl, data_ctrl, model_ctrl.key("home").id)
    data_ctrl.qpos[-7:-4] = np.array([0.55 + np.random.uniform(-0.1, 0.1), 0.15 + np.random.uniform(-0.1, 0.1), 0.15])
    data_ctrl.qpos[-4:] = R.from_euler('xyz', [0, 0, np.random.uniform(-90, 90)], degrees=True).as_quat(scalar_first=True)
    model_ctrl.body_pos[hole_id] = np.array([0.65 + np.random.uniform(-0.05, 0.05), 0.4 + np.random.uniform(-0.05, 0.05), 0.25])
    model_ctrl.body_quat[hole_id] = R.from_euler('xyz', [np.random.uniform(0, -60), np.random.uniform(-20, 20), 
                                                         np.random.uniform(-20, 20)], degrees=True).as_quat(scalar_first=True)
    
    # Initialize mink configuration
    configuration.update(data_ik.qpos)
    posture_task.set_target_from_configuration(configuration)
    data_ctrl.ctrl[:] = configuration.q
    
    state_freq = 100
    img_freq = 10
    data_logger = BCDataLogger(filename=f"./bc_data/peg_insertion/data_{len(os.listdir('./bc_data/peg_insertion/'))}.hdf5", state_freq=state_freq, img_freq=img_freq)
    renderer = mujoco.Renderer(model_ctrl, height = 240, width = 320)
    
    step = 0
    rate = RateLimiter(frequency=1000.0, warn=False)
    render_freq = 60.0  # Target Hz for VR streaming
    render_interval = 1.0 / render_freq
    prev_time = 0.0
    
    start_dc = False
    try:
        with mujoco.viewer.launch_passive(model_ctrl, data_ctrl, show_left_ui=True, show_right_ui=False, key_callback=key_callback) as viewer:
            viewer.cam.azimuth = 70
            viewer.cam.elevation = -30
            viewer.cam.distance = 2.3
            while True:
                step_timestamp = time.monotonic()
                if not data_queue.empty():
                    start_dc = True
                    current_data = data_queue.get_nowait()
                    T_f_ps = current_data['landmarks']
                    T_a_mw = current_data['arm']
                    
                    end_effector_task.set_target(T_a_mw)
                    T_f_mws = set_finger_tasks(finger_tasks, T_f_ps, T_a_mw, selected_ids)
                    set_mocap(model_ctrl, data_ctrl, T_f_mws, T_a_mw, f_b_mocap_tgts)
                    
                    vel = mink.solve_ik(configuration, tasks, model_ctrl.opt.timestep, solver, limits=limits)
                    configuration.integrate_inplace(vel, model_ctrl.opt.timestep)

                    data_ctrl.ctrl[:] = configuration.q
                
                mujoco.mj_step(model_ctrl, data_ctrl)
                if start_dc:
                    print(step / state_freq)

                    step += 1
                    img1, img2 = None, None
                    # print(data_ctrl.site(arm_ee_id).xpos, 
                    #       R.from_matrix(data_ctrl.site(arm_ee_id).xmat.reshape((3,3))).as_quat(scalar_first=True),
                    #       [T_f_p.wxyz_xyz[4:] for T_f_p in T_f_ps],data_ctrl.qpos.copy(),
                    #     data_ctrl.qvel.copy(),[data_ctrl.xfrc_applied[id].copy() for id in f_b_ids],
                    #     data_ctrl.body(peg_id).xpos.copy(),
                    #     data_ctrl.body(peg_id).xquat.copy(),
                    #     # hole position xyz and quat
                    #     data_ctrl.body(hole_id).xpos.copy(),
                    #     data_ctrl.body(hole_id).xquat.copy(),)
                    state = np.concatenate([
                        # arm ee pos and quat
                        *data_ctrl.site(arm_ee_id).xpos.tolist(),
                        R.from_matrix(data_ctrl.site(arm_ee_id).xmat.reshape((3,3))).as_quat(scalar_first=True).tolist(),
                        # hand finger bone pos
                        *[T_f_p.wxyz_xyz[4:] for T_f_p in T_f_ps], # for se3 in T_f_p],
                        # qvel, qpos
                        data_ctrl.qpos.copy(),
                        data_ctrl.qvel.copy(),
                        # hand forces on each body
                        *[data_ctrl.xfrc_applied[id].copy() for id in f_b_ids],
                        # camera images 1, 2,
                        # peg position xyz and quat
                        data_ctrl.body(peg_id).xpos.copy(),
                        data_ctrl.body(peg_id).xquat.copy(),
                        # hole position xyz and quat
                        data_ctrl.body(hole_id).xpos.copy(),
                        data_ctrl.body(hole_id).xquat.copy(),
                    ])
                    action = data_ctrl.ctrl.copy()
                    
                    curr_time = time.monotonic()
                    if (curr_time - prev_time) >= render_interval:
                        prev_time = curr_time

                        renderer.update_scene(data_ctrl, camera='cam1')
                        img1 = renderer.render()
                        renderer.update_scene(data_ctrl, camera='cam2')
                        img2 = renderer.render()
                    
                    data_logger.log_step(step_timestamp, state, img1, img2, action, 1/state_freq)

                if done_dc:
                    break
                
                rate.sleep()
                viewer.sync()
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving data...")
        data_logger.save_to_hdf5()

if __name__ == "__main__":
    bc_runner()