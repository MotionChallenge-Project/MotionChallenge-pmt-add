"""Compare against pinned PMT observation terms and ActorCritic, not its mjlab demo.

uv run validate_inference.py --source /path/to/PMT --isaac-math /path/to/math.py
  --isaac-observations /path/to/observations.py
  --isaac-articulation /path/to/articulation_data.py
  --weights /path/to/add_multimotion_flat.pt --output /outside/git/parity.json

Isaac math: source/isaaclab/isaaclab/utils/math.py at native-sources.json commit.
Executes the original function/class AST without simulator imports or JIT
decorators; no Isaac simulator is needed. This checks inference, not physics.
"""
import argparse
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from functools import reduce
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from scipy.spatial.transform import Rotation as R
from adapter import PMTADD
from motion_challenge.sdk import BodyState, StatePacket, ReferenceFrame


def source_definitions(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('missing native definitions')
    for n in nodes:
        # Registry and JIT decoration do not change the evaluated formulas.
        n.decorator_list = []
    prefix = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), str(path), 'exec'), scope)


def validate_control_config(source, root, adapter, pinned):
    """Re-evaluate the pinned author's config, using inert config containers."""
    class Config(NS):
        pass
    Config.InitialStateCfg = Config
    Config.JointDriveCfg = Config
    Config.PDGainsCfg = Config
    scope = {'ASSET_DIR': '', 'ArticulationCfg': Config, 'ImplicitActuatorCfg': Config,
             'sim_utils': NS(**{n:Config for n in ['UrdfFileCfg','UsdFileCfg','RigidBodyPropertiesCfg',
                        'ArticulationRootPropertiesCfg','UrdfConverterCfg',
                        'CollisionPropertiesCfg','PreviewSurfaceCfg']})}
    tree = ast.parse((source/'pmt_tasks/robots/g1.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))
                 and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                          and isinstance(n.value.func,ast.Name) and n.value.func.id=='print')]
    exec(compile(tree, 'native-robot-config', 'exec'),scope)
    robot = scope['G1_CYLINDER_CFG']
    def match(values,name,default=None):
        if not isinstance(values,dict): return values
        hits = [v for pattern,v in values.items() if re.fullmatch(pattern,name)]
        if not hits and default is not None: return default
        if len(hits)!=1: raise ValueError('ambiguous native config: '+name)
        return hits[0]
    defaults=[];scales=[];kps=[];kds=[]
    for name in adapter.names:
        matches=[a for a in robot.actuators.values()
                 if any(re.fullmatch(pattern,name) for pattern in a.joint_names_expr)]
        assert len(matches)==1
        actuator=matches[0]
        defaults.append(match(robot.init_state.joint_pos,name,0.))
        scales.append(match(scope['G1_ACTION_SCALE'],name))
        kps.append(match(actuator.stiffness,name));kds.append(match(actuator.damping,name))
    for actual,expected in [(adapter.default,defaults),(adapter.scale,scales),
                            (adapter.kp,kps),(adapter.kd,kds)]:
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
    xml=root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.xml'
    assert hashlib.sha256(xml.read_bytes()).hexdigest()==pinned['canonical_robot_xml_sha256']
    pelvis=ET.parse(xml).find('.//body[@name="pelvis"]/inertial')
    np.testing.assert_array_equal(adapter.cfg['canonical_root_com_offset_m'],
                                  [float(v) for v in pelvis.attrib['pos'].split()])
    return {'native_pd_default_action_scale':True,'canonical_pelvis_com':True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--isaac-math', type=Path, required=True)
    p.add_argument('--isaac-observations', type=Path, required=True)
    p.add_argument('--isaac-articulation', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    pinned = json.loads((root/'native-sources.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in pinned['files'].items():
        if sha(args.source/name) != expected:
            raise ValueError('native source mismatch: '+name)
    if sha(args.isaac_math) != pinned['isaaclab_math_sha256']:
        raise ValueError('Isaac Lab math checksum mismatch')
    for path, key in [(args.isaac_observations, 'isaaclab_observations_sha256'),
                      (args.isaac_articulation, 'isaaclab_articulation_sha256')]:
        if sha(path) != pinned[key]:
            raise ValueError('Isaac Lab velocity definition checksum mismatch')
    runtime = json.loads((root/'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    adapter = PMTADD(root, runtime)
    config_checks = validate_control_config(args.source,root,adapter,pinned)
    config_tree = ast.parse((args.source/'pmt_tasks/env_cfgs/add/add_multimotion.py').read_text())
    constants = [n for n in config_tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)
                 and n.target.id in ('ADD_KEY_BODY_NAMES','ADD_ANCHOR_BODY','TRACKED_BODY_NAMES')]
    config_scope = {'List':list}
    exec(compile(ast.Module(body=constants,type_ignores=[]),'native-add-body-config','exec'),config_scope)
    assert adapter.cfg['body_names']==config_scope['TRACKED_BODY_NAMES']
    assert adapter.cfg['anchor_body']==config_scope['ADD_ANCHOR_BODY']
    network_config = (args.source/'configs/network/mlp.yaml').read_text()
    for role in ('actor','critic'):
        dims = re.search(role+r'_hidden_dims: (\[[^\n]+\])',network_config).group(1)
        assert ast.literal_eval(dims)==[1024,512,512,256]
    config_checks['native_add_body_order_and_network']=True

    scope = {'torch': torch, 'nn': nn, 'np': np, 'reduce': reduce, 'Normal': Normal,
             'resolve_nn_activation': lambda name: {'elu': nn.ELU}[name]()}
    base = args.source/'motion_tracking_rl/networks'
    source_definitions(base/'layers/normalization.py', ['EmpiricalNormalization'], scope)
    source_definitions(base/'layers/mlp.py', ['MLP'], scope)
    source_definitions(base/'actor_critic.py', ['ActorCritic'], scope)
    native = scope['ActorCritic']({'policy': torch.zeros(1,205), 'critic': torch.zeros(1,205)},
        {'policy': ['policy'], 'critic': ['critic']}, 29,
        actor_hidden_dims=[1024,512,512,256], critic_hidden_dims=[1024,512,512,256],
        actor_obs_normalization=True, critic_obs_normalization=True, activation='elu').eval()
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    native.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert native.actor_obs_normalizer.eps == adapter.eps == .01
    source_definitions(args.isaac_math, ['normalize', 'matrix_from_quat', 'quat_conjugate',
        'quat_inv', 'quat_mul', 'quat_apply', 'quat_apply_inverse', 'subtract_frame_transforms'], scope)
    source_definitions(args.source/'pmt_tasks/mdp/observations.py', ['tracking_obs'], scope)
    # Execute the actual Isaac Lab aliases and observation function to check
    # the COM-versus-link-origin distinction, not only the coordinate rotation.
    data_class = next(n for n in ast.parse(args.isaac_articulation.read_text()).body
                      if isinstance(n,ast.ClassDef) and n.name == 'ArticulationData')
    methods = [n for n in data_class.body if isinstance(n,ast.FunctionDef)
               and n.name in ['root_com_lin_vel_b','root_lin_vel_b']]
    for n in methods: n.decorator_list = []
    scope['math_utils'] = NS(**{k:scope[k] for k in ['quat_apply_inverse']})
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+methods,
                           type_ignores=[]), str(args.isaac_articulation), 'exec'),scope)
    NativeData = type('NativeData',(),{n.name:property(scope[n.name]) for n in methods})
    scope['SceneEntityCfg'] = lambda name:NS(name=name)
    source_definitions(args.isaac_observations,['base_lin_vel'],scope)
    rng = np.random.default_rng(261023)
    errors = dict(observation=0., normalized=0., network=0., action=0., motor_target=0., reset_history=0.)
    def tensor(x): return torch.tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)
    def quat(x): return tensor(np.asarray(x)[..., [3,0,1,2]])
    def body(name):
        return BodyState(name, tuple(rng.normal(size=3)), tuple(R.random(random_state=rng).as_quat()),
                         tuple(rng.normal(size=3)), tuple(rng.normal(size=3)))
    def error(key, a, b):
        v = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[key] = max(errors[key], v)
    # Randomize transport joint order to verify both permutation directions.
    names = tuple(np.array(adapter.names)[rng.permutation(29)])
    for reset in range(4):
        native_last = torch.zeros(1,29)
        for step in range(40):
            bodies = tuple(body(n) for n in adapter.cfg['body_names'])
            refs = tuple(body(n) for n in adapter.cfg['body_names'])
            root_body = body('pelvis')
            state = StatePacket(step*.02, names, tuple(rng.normal(size=29)), tuple(rng.normal(size=29)),
                root_body.position_world_m, root_body.orientation_xyzw,
                root_body.linear_velocity_world_m_s, root_body.angular_velocity_world_rad_s, bodies)
            ref = ReferenceFrame(state.time_s, state.time_s, tuple(rng.normal(size=29)),
                refs[0].position_world_m, refs[0].orientation_xyzw, tuple(rng.normal(size=29)),
                refs[0].linear_velocity_world_m_s, refs[0].angular_velocity_world_rad_s, refs)
            if step == 0:
                adapter.reset(state, (ref,), reset)
                error('reset_history', adapter.last, native_last.numpy()[0])
            actual_anchor = bodies[adapter.cfg['body_names'].index('torso_link')]
            ref_anchor = refs[adapter.cfg['body_names'].index('torso_link')]
            command = NS(cfg=NS(body_names=adapter.cfg['body_names']),
                robot_anchor_pos_w=tensor(actual_anchor.position_world_m), robot_anchor_quat_w=quat(actual_anchor.orientation_xyzw),
                anchor_pos_w=tensor(ref_anchor.position_world_m), anchor_quat_w=quat(ref_anchor.orientation_xyzw),
                robot_body_pos_w=tensor([b.position_world_m for b in bodies]),
                robot_body_quat_w=quat([b.orientation_xyzw for b in bodies]))
            env = NS(num_envs=1, command_manager=NS(get_term=lambda _:command))
            data = NativeData()
            data.root_link_quat_w = quat(state.root_orientation_xyzw)
            offset_w = scope['quat_apply'](data.root_link_quat_w,tensor(adapter.cfg['canonical_root_com_offset_m']))
            data.root_com_lin_vel_w = tensor(state.root_linear_velocity_world_m_s)+torch.cross(
                tensor(state.root_angular_velocity_world_rad_s),offset_w,dim=-1)
            env.scene = {'robot':NS(data=data)}
            obs = torch.cat([
                tensor(np.asarray(ref.joint_position_rad)[adapter.order]),
                tensor(np.asarray(ref.joint_velocity_rad_s)[adapter.order]),
                *[scope['tracking_obs'](env,'motion',k) for k in
                  ['motion_anchor_pos','motion_anchor_ori','robot_body_pos','robot_body_ori']],
                scope['base_lin_vel'](env),
                scope['quat_apply_inverse'](quat(state.root_orientation_xyzw),tensor(state.root_angular_velocity_world_rad_s)),
                tensor(np.asarray(state.joint_position_rad)[adapter.order])-tensor(adapter.default),
                tensor(np.asarray(state.joint_velocity_rad_s)[adapter.order]), native_last], dim=1)
            actual_obs = adapter.observations(state, (ref,))
            error('observation', actual_obs, obs.numpy()[0])
            with torch.inference_mode():
                native_action = native.act_inference({'policy':obs})
                # Identical raw observations isolate checkpoint/network parity.
                network_only = native.act_inference({'policy':torch.from_numpy(actual_obs)[None]})
                error('network', adapter.infer(actual_obs), network_only.numpy()[0])
                error('normalized', (torch.from_numpy(actual_obs)[None]-adapter.mean)/(adapter.std+adapter.eps),
                      native.actor_obs_normalizer(obs))
            control = adapter.act(state, (ref,))
            error('action', adapter.last, native_action.numpy()[0])
            expected_target = tensor(adapter.default)+tensor(adapter.scale)*native_action
            error('motor_target', control.joint_position_target_rad, expected_target.numpy()[0][adapter.inverse])
            native_last = native_action.clone()
    # Float64 scipy frame operations vs native float32 quaternion math explain
    # small observation rounding; identical-observation network output is exact.
    limits = dict(observation=2e-4, normalized=2e-3, network=1e-6,
                  action=2e-4, motor_target=2e-4, reset_history=0.)
    report = {'passed':all(errors[k]<=limits[k] for k in errors), 'cases':160, 'resets':4,
        'max_abs_error':errors, 'limits':limits, 'checkpoint_sha256':runtime['weights_sha256'],
        'source_commit':pinned['pmt_commit'], 'isaaclab_math_commit':pinned['isaaclab_commit'],
        'normalizer_epsilon':adapter.eps, 'torch_version':torch.__version__,
        'config_checks':config_checks,
        'scope':'Original PMT observation functions and ActorCritic AST plus original IsaacLab math functions. Simulator imports and decorators omitted; inference and named-joint mapping only, not training-physics equivalence.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed'], 'native parity failed'


if __name__ == '__main__':
    main()
