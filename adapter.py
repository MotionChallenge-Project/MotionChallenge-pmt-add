"""PMT ADD: original PyTorch checkpoint, native 205D privileged observations."""
import json
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import vector
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


class PMTADD:
    def __init__(self, root, runtime):
        cfg_path = root / runtime['inference_config']
        weights = root / runtime['weights']
        if sha256_file(cfg_path) != runtime['inference_config_sha256']:
            raise ValueError('config checksum mismatch')
        if sha256_file(weights) != runtime['weights_sha256']:
            raise ValueError('checkpoint checksum mismatch')
        self.cfg = json.loads(cfg_path.read_text())
        self.names = self.cfg['policy_joint_names']
        self.default = vector(self.cfg['default_joint_pos'], self.names).astype(np.float32)
        self.scale = vector(self.cfg['action_scale'], self.names).astype(np.float32)
        self.kp = vector(self.cfg['joint_kp'], self.names)
        self.kd = vector(self.cfg['joint_kd'], self.names)
        torch.set_num_threads(1)
        sd = torch.load(weights, map_location='cpu', weights_only=True)['model_state_dict']
        layers = []
        for a, b in zip([205, 1024, 512, 512, 256], [1024, 512, 512, 256, 29]):
            layers.extend([nn.Linear(a, b), nn.ELU()])
        self.actor = nn.Sequential(*layers[:-1]).eval()
        self.actor.load_state_dict({k.removeprefix('actor.'): v for k, v in sd.items() if k.startswith('actor.')}, strict=True)
        self.mean = sd['actor_obs_normalizer._mean']
        self.std = sd['actor_obs_normalizer._std']
        # Native EmpiricalNormalization default; the author's mjlab demo uses a
        # different epsilon. Match ActorCritic.act_inference, not that demo.
        self.eps = 0.01

    def reset(self, state, reference, seed):
        self.order = [state.joint_names.index(n) for n in self.names]
        self.inverse = [self.names.index(n) for n in state.joint_names]
        self.last = np.zeros(29, dtype=np.float32)

    def observations(self, state, reference):
        bodies = {b.name: b for b in state.bodies}
        ref_bodies = {b.name: b for b in reference[0].bodies}
        anchor = bodies[self.cfg['anchor_body']]
        target = ref_bodies[self.cfg['anchor_body']]
        inv_anchor = R.from_quat(anchor.orientation_xyzw).inv()
        root_inverse = R.from_quat(state.root_orientation_xyzw).inv()
        # Isaac Lab base_lin_vel is root COM velocity expressed in the link
        # frame. The benchmark packet exposes link-origin velocity instead.
        com_offset_world = root_inverse.inv().apply(self.cfg['canonical_root_com_offset_m'])
        com_velocity = np.asarray(state.root_linear_velocity_world_m_s) + np.cross(
            state.root_angular_velocity_world_rad_s, com_offset_world)
        body_pos = [bodies[n].position_world_m for n in self.cfg['body_names']]
        body_rot = R.from_quat([bodies[n].orientation_xyzw for n in self.cfg['body_names']])
        obs = np.concatenate([
            np.asarray(reference[0].joint_position_rad)[self.order],
            np.asarray(reference[0].joint_velocity_rad_s)[self.order],
            inv_anchor.apply(np.asarray(target.position_world_m) - anchor.position_world_m),
            (inv_anchor * R.from_quat(target.orientation_xyzw)).as_matrix()[:, :2].reshape(-1),
            inv_anchor.apply(np.asarray(body_pos) - anchor.position_world_m).reshape(-1),
            (inv_anchor * body_rot).as_matrix()[:, :, :2].reshape(-1),
            root_inverse.apply(com_velocity),
            root_inverse.apply(state.root_angular_velocity_world_rad_s),
            np.asarray(state.joint_position_rad)[self.order] - self.default,
            np.asarray(state.joint_velocity_rad_s)[self.order],
            self.last,
        ]).astype(np.float32)
        if obs.shape != (205,) or not np.isfinite(obs).all():
            raise ValueError('invalid PMT observation')
        return obs

    @torch.inference_mode()
    def infer(self, obs):
        return self.actor((torch.from_numpy(obs).reshape(1, 205) - self.mean) / (self.std + self.eps))[0].numpy()

    def act(self, state, reference):
        action = self.infer(self.observations(state, reference))
        if not np.isfinite(action).all():
            raise ValueError('nonfinite PMT action')
        self.last = action.copy()
        # Native action manager has no clipping; limits remain in the kernel.
        target = self.default + self.scale * action
        def ordered(value):
            return tuple(float(v) for v in value[self.inverse])
        return ControlCommand(ordered(target), (0.,) * 29, ordered(self.kp), ordered(self.kd), (0.,) * 29)

    def close(self):
        self.actor = None


def create(root, runtime):
    return PMTADD(root, runtime)
