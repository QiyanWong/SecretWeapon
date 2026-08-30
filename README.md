# ⚡ SecretWeapon - 2D Action Game AI Vision & Intelligent Automation Framework

An end-to-end computer vision and intelligent decision-making automation framework designed for 2D side-scrolling action games. It integrates a **PyQt5 + DirectInput** real-time control console, **Ultralytics YOLOv8** object detection engine, **XML Foothold Topological A\* Pathfinding System**, **God-View Global Panoramic Tracking**, and an **Intelligent Combat Decision Engine**.

---

## 🌟 Key Features

1. **Real-Time Multi-Target Object Detection (YOLOv8)**
   - High-throughput, low-latency detection of player states (left, right, climbing), various monster categories, ropes, ladders, and portals.
   - Built-in character position memory and stabilization to handle damage invincibility blinking and temporary occlusion.

2. **Topological A\* Pathfinding & Physics Action Planning (Foothold A\* Engine)**
   - Directly parses map physics geometry (extracting physical platforms, footholds, and ladder/rope topologies).
   - Seamless cross-platform route planning: horizontal running, hanging-ladder jump-climbing (`Jump + UP`), vertical ascension, and drop-down jumps.
   - Supports visual waypoint recording, loop patrol route generation, and dynamic replanning.

3. **God-View Minimap Tracking & World Coordinate Mapping**
   - Dynamic template matching to track character position on the minimap in real time.
   - Real-time panoramic display of map physics footholds, recorded patrol nodes, and character trajectory.

4. **Intelligent Combat & Skill Decision Engine**
   - **100% Guaranteed Target Facing**: Pre-taps directional keys before skill execution based on relative target coordinates to guarantee frontal hits.
   - **Independent Skill Cooldowns**: Fully decoupled attack cadence timers for normal attacks and AoE skills.
   - **Directional & Omni-directional AoE Modes**: Supports forward-facing cluster detection as well as surrounding area-of-effect triggers.
   - **Unidirectional Silence Debounce Timer**: Instant 0-delay transition into combat upon encountering monsters; requires continuous silence duration before smoothly returning to patrol mode.

5. **Automated Buff & Consumables Manager**
   - Independent countdown timers for multi-skill buff upkeep and automatic re-casting.

---

## 📁 Directory Structure

```
SecretWeapon/
├── a_star_pathfinder.py          # [Pathfinding] Topological A* search & action sequence generator
├── map_parser.py                 # [Map Parser] XML foothold and ladder/rope geometry parser
├── decision_engine.py            # [Decision Engine] Finite State Machine (FSM), combat arbitration
├── game_controller.py            # [Input Controller] Hardware-level DirectInput key simulation
├── minimap_tracker.py            # [Minimap Tracking] God-View panoramic mapping & route manager
├── yolo_detector.py              # [Main Application] PyQt5 control console & live inference GUI
├── gui_collector.py              # [Data Collection] Multi-mode game screen capture tool
├── train_yolo.py                 # [Training Pipeline] Automated dataset split & YOLOv8 training
├── generate_synthetic_dataset.py # [Data Synthesis] Synthetic image generation & augmentation
├── requirements.txt              # Python dependency manifest
├── dataset/                      # Configuration files & templates
│   ├── combat_config.json        # Persistent combat & skill configurations
│   └── player_template.png       # Minimap character tracking template
├── map/                          # Map topology XML files
└── run_detector.bat              # One-click launch script
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Python 3.10+ is recommended:

```bash
git clone https://github.com/QiyanWong/SecretWeapon.git
cd SecretWeapon
pip install -r requirements.txt
```

### 2. Launch Main Console

Simply double-click:
```bash
run_detector.bat
```
Or launch via command line:
```bash
python yolo_detector.py
```

---

## 🎮 User Guide

1. **Select Game Window**: Choose the target game window from the GUI dropdown menu.
2. **Load Map XML (Optional)**: Select the corresponding map XML file from the pathfinding panel to enable physics-aware topological A* navigation.
3. **Configure Combat & Buff Skills**:
   - Set keybindings and independent attack intervals for basic attack and AoE skill;
   - Set monster count threshold and Single-Direction / Bi-Directional detection mode;
   - Add automated buff keys and their respective cooldown timers.
4. **Start Automation**: Click **`⚡ Start Bot / 打怪`** to begin fully automated hunting and patrol.

---

## ⚠️ Disclaimer

This project is developed exclusively for educational, computer vision research, and algorithm study purposes. Do not use this software for any commercial purposes or in violation of any game's End User License Agreement (EULA) or Terms of Service.
