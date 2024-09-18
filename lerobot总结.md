# lerobot总结


# 1、项目结构：
```Plain Text
.
├── benchmarks           # 数据视频流的处理
|   └── video   
├── examples             # 包含演示示例，从这里开始了解LeRobot
|   └── advanced         # 包含更多高级示例，包含加载数据、评测、训练的md文档及用法，参考性比较高
├── lerobot
|   ├── common           # 包含类和实用工具
|   |   ├── datasets       # 操作hf数据集以及处理视频数据，包含各种人类演示数据集：aloha, pusht, xarm
|   |   ├── envs           # 工厂类，根据入参的配置，获取并动态返回仿真环境：aloha, pusht, xarm
|   |   ├── policies       # 各种策略：act, diffusion, tdmpc
|   |   ├── robot_devices  # 各种真实设备：dynamixel电机、opencv相机、koch机器人
|   |   └── utils          # 各种实用工具：
|   ├── configs          # 包含可在命令行中覆盖的所有选项的hydra yaml文件
|   |   ├── default.yaml   # 默认选择，加载pusht环境和扩散策略
|   |   ├── env            # 各种仿真环境及其数据集：aloha.yaml, pusht.yaml, xarm.yaml
|   |   └── policy         # 各种策略：act.yaml, diffusion.yaml, tdmpc.yaml
|   └── scripts          # 包含通过命令行执行的函数
|       ├── eval.py                 # 加载策略并在环境中评估
|       ├── train.py                # 通过模仿学习和/或强化学习训练策略
|       ├── control_robot.py        # 遥控真实机器人、记录数据、运行策略
|       ├── push_dataset_to_hub.py  # 将数据集转换为LeRobot数据集格式并上传到Hugging Face hub
|       └── visualize_dataset.py    # 加载数据集并渲染其演示
├── outputs               # 包含脚本执行结果：日志、视频、训练模型检查点、评测结果
└── tests  
```


# 2、数据格式
aloha采集的数据格式：

* 示例：episode\_0.hdf5  压缩的文件

```python
import h5py

data_dir = "/Users/xx/Downloads"
data_file = f"{data_dir}/aloha_mobile_shrimp_truncated/episode_2.hdf5"

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + f': {val.shape} * {val[0].dtype}')
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '└── ' + key + f': {val.shape} * {val[0].dtype}')
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')

with h5py.File(data_file, 'r') as hf:
    print(f'\n{data_file} : \n')
    h5_tree(hf)
    print('\n')
```
输出结果

```Plain Text
episode_2.hdf5 文件内部结构

└── action: (3750, 14) * float32
└── base_action: (3750, 2) * float32
└── compress_len: (3, 4500) * float32
└── observations
    └── effort: (3750, 14) * float32
    ├── images
    │   └── cam_high: (3750, 21167) * uint8
    │   └── cam_left_wrist: (3750, 21167) * uint8
    │   └── cam_right_wrist: (3750, 21167) * uint8
    └── qpos: (3750, 14) * float32
    └── qvel: (3750, 14) * float32

运行结果表明，这段真实训练数据，记录了 3750 个时刻的系统状态，系统状态中包含 action，每一个 action 包含 14 个关节的参数。
```


umi采集的数据格式：

* 示例：data.zarr  文件夹，以zarr结尾

```Plain Text
data.zarr
├── data           # 数据视频流的处理
|   └── camera0_rgb  
|   └── robot0_demo_end_pose
|   └── robot0_demo_start_pose
|   └── robot0_eef_pos
|   └── robot0_eef_rot_axis_angle
|   └── robot0_gripper_width 
├── meta
|   └── episode_ends
```
lerobot的标准数据集

具体数据集目录：lerobot/tests/data/lerobot

* lerobot/aloha\_static\_coffee
* lerobot/pusht



```Plain Text
数据集属性：
  ├ hf_dataset：一个 Hugging Face 数据集（由 Arrow/parquet 支持）。典型特征示例：
  │  ├ observation.images.cam_high (VideoFrame)：
  │  │   VideoFrame = {'path': mp4 视频路径, 'timestamp' (float32): 视频中的时间戳}
  │  ├ observation.state (float32 列表)：例如机械臂关节位置
  │  ... (更多观察)
  │  ├ action (float32 列表)：例如机械臂关节目标位置
  │  ├ episode_index (int64)：此样本的剧集索引
  │  ├ frame_index (int64)：此样本在剧集中的帧索引；每个剧集从 0 开始
  │  ├ timestamp (float32)：剧集中的时间戳
  │  ├ next.done (bool)：表示剧集结束；每个剧集的最后一帧为 True
  │  └ index (int64)：整个数据集中的通用索引
  ├ episode_data_index：包含每个剧集的起始和结束索引的两个张量
  │  ├ from (1D int64 张量)：每个剧集的第一帧索引 — 形状 (num episodes,) 从 0 开始
  │  └ to：(1D int64 张量)：每个剧集的最后一帧索引 — 形状 (num episodes,)
  ├ stats：数据集中每个特征的统计信息（最大值、平均值、最小值、标准差）字典，例如
  │  ├ observation.images.cam_high：{'max': 具有相同维度数的张量（例如图像为 `(c, 1, 1)`，状态为 `(c,)`），等}
  │  ...
  ├ info：数据集元数据字典
  │  ├ codebase_version (str)：用于跟踪创建数据集的代码库版本
  │  ├ fps (float)：数据集记录/同步的每秒帧数
  │  ├ video (bool)：指示帧是否编码为 mp4 视频文件以节省空间或存储为 png 文件
  │  └ encoding (dict)：如果是视频，这记录了用于编码视频的 ffmpeg 主要选项
  ├ videos_dir (Path)：存储/访问 mp4 视频或 png 图像的位置
  └ camera_keys (字符串列表)：在数据集返回的项目中访问相机特征的键（例如 `["observation.images.cam_high", ...]`）

```
aloha及umi的数据集上传及转换

```bash
# 登录hf：  
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential 替换自己的hf 有写入权限的token
# 执行上传脚本
python lerobot/scripts/push_dataset_to_hub.py \
    --raw-dir data/pusht_raw \
    --raw-format pusht_zarr \
    --repo-id lerobot/pusht \
    --local-dir data/lerobot/pusht \
    --push-to-hub True \
    --video True

这个命令将：
• 从 data/pusht_raw 加载 pusht_zarr 格式的原始数据集。
• 将转换后的数据集保存到 data/lerobot/pusht。
• 将数据集上传到 Hugging Face Hub 上的 lerobot/pusht 仓库。
• 将每个 episode 转换为 mp4 视频格式。
```


push\_dataset\_to\_hub 参数说明

```bash
基本选项
• -h, --help: 显示帮助信息并退出。

必需选项
• --raw-dir RAW_DIR: 原始数据集所在的目录。例如，data/aloha_mobile_chair_raw 或 data/pusht_raw。
• --raw-format RAW_FORMAT: 原始数据集的类型。例如，pusht_zarr, umi_zarr, aloha_hdf5, xarm_pkl, dora_parquet, openx_rlds。
• --repo-id REPO_ID: 在 Hugging Face 上的数据集仓库标识符，形式为用户名/数据集名，例如 lerobot/pusht, cadene/aloha_sim_insertion_human。

可选选项
• --local-dir LOCAL_DIR: 当提供了这个选项时，会将转换后的数据集写入到指定的本地目录中。例如，data/lerobot/aloha_mobile_chair。
• --push-to-hub PUSH_TO_HUB: 上传到 Hugging Face Hub。如果设置为 True 或 1，则会将数据集上传到 Hugging Face Hub。
• --fps FPS: 视频采集的帧率。如果不提供，默认使用代码中指定的默认值。
• --video VIDEO: 将每个原始数据集中的episode转换为mp4视频格式。这可以显著减少磁盘空间的使用并加快训练时的加载速度。
• --batch-size BATCH_SIZE: 用于计算数据集统计信息的 DataLoader 的批量大小。
• --num-workers NUM_WORKERS: 用于计算数据集统计信息的 DataLoader 的进程数。
• --episodes [EPISODES ...]: 如果提供了这个选项，只转换指定的episode。例如，--episodes 2 3 4。这在测试代码时非常有用。
• --force-override FORCE_OVERRIDE: 如果设置为 1，则会移除已存在的输出目录。默认情况下，如果输出目录已存在，会抛出 ValueError 异常。
• --resume RESUME: 如果设置为 1，则恢复之前的运行。
• --cache-dir CACHE_DIR: 用于存储在创建数据集过程中生成的临时视频和图像的目录。
• --tests-data-dir TESTS_DATA_DIR: 如果提供了这个选项，会将测试工件保存到指定的目录中。例如，--tests-data-dir tests/data 会保存到 tests/data/{--repo-id}。

使用示例
假设你想将位于 data/pusht_raw 的 pusht_zarr 类型的数据集转换并上传到 Hugging Face Hub 上的一个名为 lerobot/pusht 的仓库，你可以这样运行脚本：
```


# 3、评测
```bash
python lerobot/scripts/eval.py \
    -p lerobot/diffusion_pusht \
    eval.n_episodes=10 \
    eval.batch_size=10

执行结果：
{'avg_sum_reward': 114.59489091393061, 'avg_max_reward': 0.9839878305460219, 'pc_success': 70.0, 'eval_s': 52.19949388504028, 'eval_ep_s': 5.219949436187744}   

分析：
1. avg_sum_reward: 平均总奖励。表示在一个或多个episode（回合）中所有步骤获得的奖励之和的平均值。数值114.59489091393061表明，在测试期间，模型获得了相当数量的累积奖励。
2. avg_max_reward: 平均最大奖励。这个指标可能指的是每个episode中最大的单步奖励的平均值。数值0.9839878305460219可能表明在每一步中，模型能够达到的最大奖励值相对较高
3. pc_success: 成功率。这个百分比（70%）通常表示在一系列试验中，成功完成任务的比例。这意味着在您的实验或测试中，有70%的情况下达到了预期目标。
4. eval_s: 这个指标可能是某种评估分数，数值52.19949388504028，但是没有上下文很难确定它的具体含义。它可能是针对特定任务或目标的得分。
5. eval_ep_s: 可能是每个episode的评估得分，或者是与eval_s相关的另一个指标，数值5.219949436187744。如果eval_s是总体评估得分，那么eval_ep_s可能是平均到每个episode上的得分。   
```
以下是评测后的一个输出结果

lerobot/outputs/eval/2024-09-05/15-51-27\_pusht\_diffusion

```bash
结构：
├── 15-51-27_pusht_diffusion           # 数据视频流的处理
|   └── eval_info.json                 # 评测结果
|   └── videos                         # 视频数据

# eval_info.json 结构：
{
  "per_episode": [
    {
      "episode_ix": 0,
      "sum_reward": 65.6264227807678,
      "max_reward": 1.0,
      "success": true,
      "seed": 100000
    },
  ......
    {
      "episode_ix": 9,
      "sum_reward": 159.26678316349086,
      "max_reward": 0.9872313189632149,
      "success": false,
      "seed": 100009
    }
  ],
  "aggregated": {
    "avg_sum_reward": 94.73332028358872,
    "avg_max_reward": 0.9900758995852886,
    "pc_success": 80.0,
    "eval_s": 52.03769087791443,
    "eval_ep_s": 5.2037691354751585
  },
  "video_paths": [
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_0.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_1.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_2.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_3.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_4.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_5.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_6.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_7.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_8.mp4",
    "outputs/eval/2024-09-05/15-51-27_pusht_diffusion/videos/eval_episode_9.mp4"
  ]
}

```


# 4、训练
```bash
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human \

```


```bash
python lerobot/scripts/train.py \
    hydra.run.dir=outputs/train/clamp_cudbe_into_cup_test_tf \
    device=cuda
    env=aloha \
    env.task=AlohaTransferCube-v0 \
    dataset_repo_id=lerobot/clamp_cudbe_into_cup_test_tf \
    policy=act \
    training.eval_freq=1000 \
    training.log_freq=25 \
    training.offline_steps=1001 \
    training.save_model=true \
    training.save_freq=1000 \
    eval.n_episodes=5 \
    eval.batch_size=5 \
    wandb.enable=false 


hydra.run.dir=outputs/train/grab_cube_240906: 设置Hydra框架的工作目录为 outputs/train/grab_cube_240906，这意味着所有的输出文件（如模型保存、日志等）都将存储在这个目录中。
不设置的话，默认按照https://github.com/huggingface/lerobot/blob/main/lerobot/configs/default.yaml的配置：
outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}_${env.name}_${policy.name}_${hydra.job.name} 存储


device=cuda: 指定使用的设备是CUDA，即使用NVIDIA的GPU进行计算。
env=aloha: 设定环境名称为 aloha。
env.task=AlohaTransferCube-v0: 指定要执行的任务环境为 AlohaTransferCube-v0。
dataset_repo_id=lerobot/grab_cube_240906: 指定了数据集的仓库ID，这可能是用来标识特定的数据集版本或来源。
policy=act: 设定策略为 act，这可能指的是某种行为策略或者是训练过程中使用的算法类型。
training.eval_freq=10000: 每10,000个训练步骤后评估一次模型性能。
training.log_freq=250: 每250个训练步骤后记录一次日志信息。
training.offline_steps=10010: 总共执行10,010个离线训练步骤。
training.save_model=true: 启用模型保存功能。
training.save_freq=10000: 每10,000个训练步骤后保存一次模型。
eval.n_episodes=50: 在每次评估中运行50个episode。
eval.batch_size=50: 在评估时使用的批次大小为50。
wandb.enable=false: 禁用Weights & Biases (wandb) 的日志记录功能。
```


首先，`lerobot/configs` 的目录结构是这样的：

```Plain Text
.
├── default.yaml
├── env
│   ├── aloha.yaml
│   ├── pusht.yaml
│   └── xarm.yaml
└── policy
    ├── act.yaml
    ├── diffusion.yaml
    └── tdmpc.yaml
```
默认default.yaml：

```python
defaults:
  - _self_
  - env: pusht
  - policy: diffusion
```


训练结果：

lerobot/outputs/train/act\_aloha\_sim\_transfer\_cube\_human

* train 后的目录根据    hydra.run.dir 参数指定，如不指定，\${now:%Y-%m-%d} 格式

```bash
(base)  lerobot/outputs/train/act_aloha_sim_transfer_cube_human# ll
drwxr-xr-x 5 root root 4096 Sep  9 01:43 ./
drwxr-xr-x 6 root root 4096 Sep  9 16:32 ../
drwxr-xr-x 9 root root 4096 Sep  9 09:03 checkpoints/
-rw-r--r-- 1 root root    0 Sep  9 00:28 default.log
drwxr-xr-x 9 root root 4096 Sep  9 09:03 eval/
drwxr-xr-x 2 root root 4096 Sep 11 18:47 .hydra/
```
训练的结果，可以根据最终的checkpoints，进行评测和推理



# 5、输出关节关节位置数据
主逻辑：lerobot/lerobot/scripts/control\_robot.py  

```python

def replay2(robot: Robot, episode: int, fps: int | None = None, root: str | None = 'data', repo_id:str | None="tests/data/lerobot/aloha_mobile_cabinet"):
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()


    logging.info("Replaying episode")
    say("Replaying episode", blocking=True)
    for idx in range(from_idx, to_idx):
        start_episode_t = time.perf_counter()

        action = items[idx]["action"]
        print("要执行的action 动作：----",action)
        #robot.send_action(action)

        # 验证张量形状
        num_joints = 6
        num_joints_joints = action.shape[0] 
        # if action.shape[0] != num_joints:
        #     raise ValueError(f"Invalid number of joints: expected {num_joints}, got {action.shape[0]}")

        # 将张量转换为关节角度
        joint_angles = action.numpy()
        print("Sending joint angles to robot:\n", joint_angles)


def teleoperate2(robot: Robot, fps: int | None = None, teleop_time_s: float | None = None):
    # TODO(rcadene): Add option to record logs
    # if not robot.is_connected:
    #     robot.connect()

    start_teleop_t = time.perf_counter()
    while True:
        start_loop_t = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
            break



if __name__ == "__main__":
    # main()

    #robot_overrides = 'cameras'
    robot_overrides = None
    robot = None
    # robot_cfg = init_hydra_config('lerobot/configs/robot/aloha.yaml', robot_overrides)
    # robot_cfg = init_hydra_config('lerobot/configs/robot/realman.yaml', robot_overrides)
    # robot = make_robot(robot_cfg)

    
    #teleoperate(robot=None,fps=None,teleop_time_s=None,teleop_time_s=None)
   
    replay2(robot,episode=0,fps=None,root="/home/tf/github/lerobot",repo_id="tests/data/lerobot/aloha_mobile_cabinet")
    # teleoperate2(robot,fps=None,teleop_time_s=None)
```


# 6、真实机器人配置：
`lerobot/configs/robot/koch.yaml`  在robot 下配置不同的真实机器人，

在项目中根据

主逻辑：lerobot/lerobot/common/robot\_devices的robot目录下的factory.py 工厂类生成不同的Robot实例manipulator.py#class ManipulatorRobot



[https://github.com/huggingface/lerobot/blob/main/examples/7\_get\_started\_with\_real\_robot.md](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md)

Getting Started with Real-World Robots
真实机器人入门
To assign indices to the motors, run this code in an interactive Python session. Replace the `port` values with the ones you identified earlier:
要为电机分配索引，请在交互式 Python 会话中运行此代码。将 `port` 值替换为您之前确定的值：

```python
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_port = "/dev/tty.usbmodem575E0031751"
follower_port = "/dev/tty.usbmodem575E0032081"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)
```