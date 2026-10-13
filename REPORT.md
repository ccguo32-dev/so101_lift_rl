# SO-101 仿真环境验证报告

- 日期：2026-10-10
- 工作目录：C:\Users\35184\Desktop\S101
- 结论一句话：MJCF 可正常加载并离屏渲染；6 个关节 / 6 个执行器全部一致；
  **gripper 关节值取最小值（-0.174533 rad = -10°）时为“闭合”，取最大值（+1.745329 rad = +100°）时为“张开”。**

---

## 1. Python 虚拟环境

| 项目 | 值 |
| --- | --- |
| 虚拟环境路径 | C:\Users\35184\Desktop\S101\.venv |
| Python 解释器 | C:\Users\35184\Desktop\S101\.venv\Scripts\python.exe |
| Python 版本 | 3.10.11 (满足 3.10 / 3.11 要求) |
| mujoco | 3.15.0 |
| gymnasium | 1.4.0 |
| numpy | 2.2.6 |
| imageio | 2.38.0（用于写 PNG） |

创建与安装命令：

    C:\Users\35184\AppData\Local\Programs\Python\Python310\python.exe -m venv C:\Users\35184\Desktop\S101\.venv
    C:\Users\35184\Desktop\S101\.venv\Scripts\python.exe -m pip install --upgrade pip
    C:\Users\35184\Desktop\S101\.venv\Scripts\python.exe -m pip install mujoco gymnasium numpy imageio

## 2. 仓库与模型文件

| 项目 | 值 |
| --- | --- |
| 仓库地址 | https://github.com/TheRobotStudio/SO-ARM100 |
| 本地克隆路径 | C:\Users\35184\Desktop\S101\SO-ARM100 |
| 提交 | a758567c3978dfeefe282ede0500085a48fe8f78 (main, “docs: list Seeed Studio and WowRobo first in kits section (#188)”, 2026-10-06) |
| 文件总数 | 241 |

SO101 相关文件（Simulation/SO101）：

| 文件 | 说明 |
| --- | --- |
| **scene.xml** | **本次加载的 MJCF 主入口**（include so101_new_calib.xml，并加地面/天空盒/灯光） |
| so101_new_calib.xml | 机器人本体 MJCF（新标定：各关节虚拟零点=行程中点） |
| so101_old_calib.xml | 机器人本体 MJCF（旧标定：虚拟零点=完全水平伸展位） |
| so101_new_calib_camera.xml | 带相机配置的 MJCF |
| joints_properties.xml | STS3215 电机默认参数（被 include） |
| so101_new_calib.urdf / so101_old_calib.urdf / so101_new_calib_camera.urdf / so101_leader_new_calib.urdf | 对应的 URDF |
| assets/*.stl | 31 个网格文件（含 13 个被引用的 STL） |

**MJCF 绝对路径（本次实际加载）：**

    C:\Users\35184\Desktop\S101\SO-ARM100\Simulation\SO101\scene.xml

关于“缺 mesh”的注意事项：该 MJCF 的编译器声明为 <compiler meshdir="assets" ...>，
网格以相对文件名引用（如 <mesh file="moving_jaw_so101_v1.stl"/>），并且所有被引用的
STL 都已随仓库放在 Simulation/SO101/assets 下。仓库里**没有** assets/meshes 这一级
目录，也不需要额外拷贝——只要保持 scene.xml 与 assets/ 的相对位置不变即可正常加载。
本次加载与渲染均无任何缺网格报错。

## 3. 关节清单（编译后模型，nq=6）

| 关节名 | 类型 | qpos 地址 | 限位 | 最小值 [rad] | 最大值 [rad] | 最小值 [deg] | 最大值 [deg] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shoulder_pan  | hinge | 0 | 有 | -1.919862 | 1.919862 | -110.00 | 110.00 |
| shoulder_lift | hinge | 1 | 有 | -1.745329 | 1.745329 | -100.00 | 100.00 |
| elbow_flex    | hinge | 2 | 有 | -1.690000 | 1.690000 |  -96.83 |  96.83 |
| wrist_flex    | hinge | 3 | 有 | -1.658063 | 1.658063 |  -95.00 |  95.00 |
| wrist_roll    | hinge | 4 | 有 | -2.743847 | 2.841206 | -157.21 | 162.79 |
| gripper       | hinge | 5 | 有 | -0.174533 | 1.745329 |  -10.00 | 100.00 |

所有关节均为 hinge（转动），共 6 个，全部带限位。

## 4. Actuator 清单（nu=6）

| 执行器名 | 传输类型 | 绑定关节 | gear | ctrlrange | forcerange |
| --- | --- | --- | --- | --- | --- |
| shoulder_pan  | position(joint) | shoulder_pan  | 1.0 | [-1.9199, 1.9199] | [-3.35, 3.35] |
| shoulder_lift | position(joint) | shoulder_lift | 1.0 | [-1.7453, 1.7453] | [-3.35, 3.35] |
| elbow_flex    | position(joint) | elbow_flex    | 1.0 | [-1.6900, 1.6900] | [-3.35, 3.35] |
| wrist_flex    | position(joint) | wrist_flex    | 1.0 | [-1.6581, 1.6581] | [-3.35, 3.35] |
| wrist_roll    | position(joint) | wrist_roll    | 1.0 | [-2.7439, 2.8412] | [-3.35, 3.35] |
| gripper       | position(joint) | gripper       | 1.0 | [-0.1745, 1.7453] | [-3.35, 3.35] |

6 个执行器均为 position 型（位置伺服），一对一映射到 6 个关节，
控制量单位与关节角一致（rad），ctrlrange 与关节 range 相同。

## 5. 夹爪方向结论（重点）

**结论：gripper 关节值取最小值时 = 闭合；取最大值时 = 张开。**

- qmin = -0.174533 rad = **-10°  → 闭合**
- qmax = +1.745329 rad = **+100° → 张开**（行程共 110°）
- 所有 6 个臂关节保持在中点姿态，只扫 gripper。

在关节范围内等间隔取 61 个点，用两个夹爪网格的顶点实测“指尖开口距离”：

| gripper 关节值 | 指尖开口距离 |
| --- | --- |
| qmin = -10.00° | 6.8 mm（端点，均值 9.1 mm） |
| q = 0°        | 19.3 mm |
| q = +45°      | 83.1 mm |
| qmax = +100.00° | 141.9 mm（端点，均值 140.4 mm） |

备注 1：同一份 `verify_so101_result.json` 里出现的两组开口值来自两条口径——`sweep.tip_aperture_mm` 是 61 个采样点的逐点端点值（首末两点即 -10°/100° 端点，6.8 → 141.9 mm），而 `gripper.tip_aperture_at_qmin/qmax` 取的是端点三点均值（9.1 → 140.4 mm），两者算法同为“两夹爪远端指尖点的距离”，差别只在端点单点 vs 端点三点平滑。
备注 2：对外统一引用 summary 口径（9.1 → 140.4 mm），逐点序列仅用于复核单调性与曲线形状。

- 随 q 增大，开口**单调增大**：60/60 步全部递增，无回退。
- 回归斜率 d(开口)/dq = +0.0719 m/rad。
- 独立交叉验证：另有脚本按 MJCF 中的位姿/四元数纯几何推算（不经过渲染器），
  得到“闭合端两颗牙齿最近约 2.3 mm、张开端约 132 mm，单调增大”，结论一致。

两点补充说明：

1. 直接取“两个夹爪网格之间的最小距离”会得到全程约 0.9–2.7 mm 的近似常数，
   这是被**根部铰链壳体**的几何主导的，不能用来判断开合；报告采用的是远端
   指尖间距（以及质心间距）作为开口度量。
2. 上游 README 特别注明：在 LeRobot 里 gripper 被表示为线性关节，0=完全闭合、
   100=完全张开，**这个 0–100 映射尚未反映到本仓库的 URDF/MJCF 里**。也就是说
   本 MJCF 的关节值区间是 [-10°, 100°]，与 LeRobot 的 [0, 100] 不是同一套数值，
   接入 LeRobot 时需要自行做映射。

## 6. 渲染图

| 文件 | 说明 |
| --- | --- |
| C:\Users\35184\Desktop\S101\renders\so101_frame.png | 整机一帧，640×480 |
| C:\Users\35184\Desktop\S101\renders\gripper_qmin.png | gripper=-10°（闭合）时的夹爪特写，640×480 |
| C:\Users\35184\Desktop\S101\renders\gripper_qmax.png | gripper=+100°（张开）时的夹爪特写，640×480 |
| C:\Users\35184\Desktop\S101\renders\verify_so101_result.json | 关节/执行器清单、扫描曲线的机器可读结果 |

两帧夹爪特写像素差异显著（约 5.9 万个像素变化，集中在夹爪区域），与数值结论一致。

## 7. 复现方式

    cd C:\Users\35184\Desktop\S101
    .\.venv\Scripts\python.exe verify_so101.py

脚本：C:\Users\35184\Desktop\S101\verify_so101.py
输出文本：C:\Users\35184\Desktop\S101\verify_output.txt

## 8. 过程备注

- 直连 GitHub 的 git clone 在接收 pack 阶段多次 “early EOF”（本机到 github.com:443 的
  git 传输很不稳定，而 raw/codeload 静态文件正常）。最终通过镜像
  https://gh-proxy.com 完成真正的 git clone 与 checkout，得到真实 .git 与提交号 a758567。
- git clone 到 Windows 需要 core.protectNTFS=false / core.longpaths=true，本次已在
  全局 git 配置中设置。
- 本次未修改仓库内任何 MJCF / XML / URDF 文件。

- `wrist_roll_follower_so101_v1.stl` 与 `geom_check\wrist_fixed.stl` 均为“独立几何
  交叉验证”子任务的产物；经全仓检索，二者不被任何 .xml / .py 引用，已统一归档到 `geom_check\`，
  仅作为可追溯的参考几何保留，不参与 verify_so101.py 与 so101_lift 的运行。
  M1 正式交付物：SO-ARM100、.venv、renders、so101_lift、geom_check、.gitignore、
  requirements.txt、README.md、verify_so101.py、verify_output.txt、REPORT.md。
