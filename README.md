# Vision-Language-Action (VLA) Intelligent Robot for Real-World Task Execution

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Iron-orange.svg)](https://docs.ros.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end Physical AI manipulation system that transcribes natural language human voice commands, perceives the 3D physical workspace using RGB-D and LiDAR sensors, orchestrates decisions across **16 logical agents**, enforces a rigorous non-AI **Independent Safety Gate**, and executes collision-free manipulation trajectories on robotic manipulators.

---

## 1. System Architecture Overview

```
                          +------------------------------------------+
                          |        Human Voice / Text Command        |
                          +------------------------------------------+
                                               |
                                               v
                                   [Agent 4: NLP Agent]
                          (Whisper STT -> NLTK Preprocessing -> SVM)
                                               |
                                               v
                                [Agent 1: Main Orchestrator] <--------+
                                (Context Retrieval & Sync)            |
                                               |                      |
                                               v                      |
                             [Agent 2: Environment Check Agent]       |
                               (Clutter & Workspace Suitability)      |
                                               |                      |
                                               v                      |
                            [Agent 3: Vision Context Prompt Agent]    |
                              (Structured Scene Context Generation)   |
                                               |                      |
                                               v                      |
                               [Agent 5: Work Splitter Agent]         |
                                (Sub-Goal Delegation)                 |
                                               |                      |
                +------------------------------+--------------------+ |
                |                              |                    | |
                v                              v                    v |
       [Agent 6: Perception]        [Agent 7: Task Planning] [Agent 8: Decision]
       (3D Object & Dest Coords)     (7 Atomic Primitives)   (Feasibility/Reach)
                |                              |                    | |
                +------------------------------+--------------------+ |
                                               |                      |
                                               v                      |
                              [Agent 9: AI Path Planning Agent]       |
                                  (Waypoints & Kinematics)            |
                                               |                      |
                                               v                      |
                                [Agent 10: Synthesizer Agent]         |
                                 (Unified Execution Plan)             |
                                               |                      |
                                               v                      |
                             [Agent 11: ROS 2 Communication Agent]    |
                                (Topics, Services, Actions)           |
                                               |                      |
                                               v                      |
                            [Agent 12: MoveIt 2 Motion Planning Agent]|
                                 (Inverse Kinematics & RRT)           |
                                               |                      |
                                               v                      |
                           +======================================+   |
                           |       INDEPENDENT SAFETY GATE        |   |
                           |  - E-Stop Monitor                    |   |
                           |  - Joint Limits & Velocity Check     |   |
                           |  - Workspace Boundary & Collisions   |   |
                           +======================================+   |
                                        /            \                |
                           (Rejected)  /              \  (Approved)   |
                                      v                v              |
                         [Agent 16: Error Agent]   [Agent 13: Hardware Agent]
                         (Replan / Recovery Loop)  (ros2_control Joint Motors)
                                      |                |
                                      +----------------+
                                                       |
                                                       v
                                         [Agent 14: Verification Agent]
                                          (Grasp Force & Placement Coords)
                                                       |
                                                       v
                                          [Agent 15: Evaluation Agent]
                                          (Accuracy Score & Metrics DB)
```

---

## 2. Complete Project Repository Tree

```
VLA-INTELLIGENT-ROBOT/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── database_config.py
│   ├── camera_config.yaml
│   ├── lidar_config.yaml
│   ├── vision_config.yaml
│   └── robot_config.yaml
│
├── agents/
│   ├── __init__.py
│   ├── main_orchestrator.py        # Agent 1
│   ├── environment_check_agent.py   # Agent 2
│   ├── vision_context_agent.py      # Agent 3
│   ├── nlp_agent.py                 # Agent 4
│   ├── work_splitter_agent.py       # Agent 5
│   ├── perception_agent.py          # Agent 6
│   ├── task_planning_agent.py       # Agent 7
│   ├── decision_agent.py            # Agent 8
│   ├── path_planning_agent.py       # Agent 9
│   ├── synthesizer_agent.py         # Agent 10
│   ├── ros2_communication_agent.py  # Agent 11
│   ├── moveit_motion_agent.py       # Agent 12
│   ├── hardware_control_agent.py    # Agent 13
│   ├── verification_agent.py        # Agent 14
│   ├── evaluation_agent.py          # Agent 15
│   └── error_handling_agent.py      # Agent 16
│
├── vision/
│   ├── __init__.py
│   ├── camera.py                   # RealSense D435i & synthetic frame capture
│   ├── object_detection.py         # YOLOv8 bounding box & CNN classes
│   ├── object_segmentation.py      # YOLOv8-Seg pixel-wise masks
│   ├── object_tracking.py          # ByteTrack multi-object tracking
│   ├── color_detection.py          # OpenCV HSV color recognition
│   ├── depth_processing.py         # RealSense depth filtering & scaling
│   ├── coordinate_estimation.py    # 2D-to-3D deprojection (intrinsics)
│   ├── camera_calibration.py       # Camera-to-base extrinsics transform
│   ├── lidar_processing.py         # 2D LiDAR range scan acquisition
│   ├── obstacle_detection.py       # Collision threat detection
│   ├── vision_pipeline.py          # Master perception coordinator
│   └── vision_state.py             # Pydantic schemas for vision state
│
├── nlp/
│   ├── __init__.py
│   ├── speech_to_text.py           # Whisper STT engine
│   ├── preprocessing.py            # NLTK pipeline (preserves spatial prepositions)
│   ├── tokenization.py             # Word & sentence tokenizers
│   ├── stemming.py                 # Porter & Snowball stemmers
│   ├── lemmatization.py            # WordNet lemmatizer with POS mapping
│   ├── pos_tagging.py              # Part-of-speech tagger
│   ├── feature_extraction.py       # TF-IDF n-gram vectorizer
│   ├── intent_classifier.py        # SVM & Logistic Regression classifiers
│   ├── entity_extraction.py        # spaCy & grammar entity parser
│   ├── command_parser.py           # Validated structured command schema
│   └── dataset/
│       └── robotic_commands.csv    # Custom command dataset
│
├── orchestration/
│   ├── __init__.py
│   ├── state.py                    # Central LangGraph AgentState model
│   ├── graph.py                    # 16-agent StateGraph pipeline
│   ├── routing.py                  # Dynamic conditional routers
│   ├── prompt_chains.py            # Sequential prompt synthesizers
│   ├── tool_registry.py            # Spatial distance & bounding tools
│   └── error_recovery.py           # Retry & fallback policies
│
├── robotics/
│   ├── __init__.py
│   ├── ros2_node.py                # ROS 2 node & topic bridges
│   ├── moveit_interface.py         # MoveIt 2 Planning Scene & MoveGroup
│   ├── tf_manager.py               # TF2 frame transformations
│   ├── inverse_kinematics.py       # Analytical 6-DOF IK solver
│   ├── trajectory_planner.py       # Quintic polynomial interpolator
│   ├── gripper_controller.py       # Parallel jaw gripper controller
│   ├── hardware_interface.py       # ros2_control hardware interface
│   └── safety/
│       ├── __init__.py
│       ├── emergency_stop.py       # E-Stop interlock monitor
│       ├── joint_limits.py         # Mechanical position & velocity audit
│       ├── collision_checker.py    # Workspace bounds & obstacle margin check
│       └── safety_gate.py          # Independent Non-AI Safety Gate
│
├── database/
│   ├── __init__.py
│   ├── connection.py               # PostgreSQL pool + SQLite memory fallback
│   ├── models.py                   # SQLAlchemy ORM declarative models
│   ├── schema.sql                  # PostgreSQL DDL migrations
│   ├── chat_history.py             # Conversation history repository
│   ├── agent_logs.py               # Agent execution telemetry repository
│   ├── task_history.py             # Task & trajectory execution records
│   └── error_logs.py               # System error & recovery records
│
├── simulation/
│   ├── __init__.py
│   ├── mock_camera.py              # Virtual tabletop RGB-D generator
│   ├── mock_lidar.py               # Virtual 2D LiDAR scanner
│   ├── mock_robot.py               # Simulated 6-DOF manipulator & gripper
│   ├── simulated_environment.py    # Virtual environment orchestrator
│   └── gazebo/
│       └── tabletop_manipulator.world
│
├── interface/
│   ├── __init__.py
│   ├── app.py                      # FastAPI web interface with UI dashboard
│   ├── notifications.py            # User banner & clarification generator
│   └── task_monitor.py             # Real-time pipeline monitor
│
├── scripts/
│   ├── initialize_database.py      # Database schema initializer
│   ├── train_intent_classifier.py  # Trains SVM on command dataset
│   ├── run_vision.py               # Standalone perception runner
│   ├── run_nlp.py                  # Standalone NLP parser runner
│   ├── run_simulation.py           # Standalone robot simulation runner
│   └── run_system.py               # Master end-to-end benchmark runner
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py              # Tests for all 16 agents
│   ├── test_vision.py              # Vision module tests
│   ├── test_nlp.py                 # NLP module tests
│   ├── test_orchestration.py       # LangGraph routing & state tests
│   ├── test_database.py            # PostgreSQL repository tests
│   ├── test_safety.py              # Independent Safety Gate tests
│   ├── test_robotics.py            # Kinematics & trajectory tests
│   └── test_end_to_end.py          # Full benchmark integration test
│
└── data/
    ├── sample_commands.csv
    └── model_artifacts/
```

---

## 3. The 16 Logical Agents

### Group A: AI Workflow and Decision-Making (Agents 1 – 10)
1. **Agent 1 – Main / Orchestrator Agent (`agents/main_orchestrator.py`)**: Central workflow coordinator. Synchronizes NLP and vision data, retrieves conversation history from PostgreSQL, checks object availability, and routes state through LangGraph.
2. **Agent 2 – Initial Environment Check Agent (`agents/environment_check_agent.py`)**: Assesses workspace suitability from vision data. Classifies status into `clean`, `manageable`, or `unsuitable`, rejecting high-clutter or unsafe scenes and issuing cleanup requests.
3. **Agent 3 – Vision Context Prompt Agent (`agents/vision_context_agent.py`)**: Translates numerical bounding boxes, depth maps, and 3D positions into rich structured linguistic scene context for planning.
4. **Agent 4 – NLP Agent (`agents/nlp_agent.py`)**: Preprocesses Whisper transcripts via NLTK, classifies intent using linear SVM, extracts target entities (objects, colors, destinations, prepositions), and emits structured schemas.
5. **Agent 5 – Work Splitter Agent (`agents/work_splitter_agent.py`)**: Decomposes validated high-level goals into sub-tasks assigned to Perception, Task Planning, Decision, and Path Planning agents.
6. **Agent 6 – Perception Agent (`agents/perception_agent.py`)**: Resolves spatial 3D grasp and placement coordinates in the robot base frame for the specific target object and destination.
7. **Agent 7 – Task Planning Agent (`agents/task_planning_agent.py`)**: Deconstructs high-level actions into 7 atomic manipulation primitives: `open_gripper` -> `approach` -> `descend` -> `grasp` -> `lift` -> `transit` -> `place` -> `release` -> `retract`.
8. **Agent 8 – Decision Agent (`agents/decision_agent.py`)**: Validates kinematic feasibility, workspace reachability, and resolves visual ambiguities (e.g. distinguishing among multiple red bottles).
9. **Agent 9 – AI Path Planning Agent (`agents/path_planning_agent.py`)**: Calculates geometric Cartesian waypoints avoiding table boundaries and sends requirements to MoveIt 2.
10. **Agent 10 – Synthesizer Agent (`agents/synthesizer_agent.py`)**: Combines task decomposition and path waypoints into a unified ROS 2 execution payload.

### Group B: ROS 2 Execution (Agents 11 – 13)
11. **Agent 11 – ROS 2 Communication Agent (`agents/ros2_communication_agent.py`)**: Bridges AI orchestration with ROS 2 Topics, Services, and Action Servers.
12. **Agent 12 – MoveIt 2 Motion Planning Agent (`agents/moveit_motion_agent.py`)**: Computes Inverse Kinematics and generates collision-free joint trajectories via MoveIt 2.
13. **Agent 13 – ros2_control Hardware Agent (`agents/hardware_control_agent.py`)**: Directly interfaces with physical or simulated motor controllers, strictly executing only trajectories authorized by the Independent Safety Gate.

### Group C: Verification, Evaluation and Error Recovery (Agents 14 – 16)
14. **Agent 14 – Verification Agent (`agents/verification_agent.py`)**: Verifies physical execution outcome via gripper grasp force and post-placement 3D camera coordinates.
15. **Agent 15 – Evaluation Agent (`agents/evaluation_agent.py`)**: Scores quantitative performance metrics (displacement accuracy %, stability, planning latency) and logs results.
16. **Agent 16 – Error Handling Agent (`agents/error_handling_agent.py`)**: Diagnoses execution failures, assesses retry limits, and triggers automatic replanning loops back to the Orchestrator or Path Planner.

---

## 4. Independent Safety Mechanism

The safety system (`robotics/safety/`) is an **independent software and hardware protection layer**, NOT an AI agent:
- **Position**: Positioned strictly between Agent 12 (MoveIt 2) and Agent 13 (ros2_control Hardware).
- **Mandatory Interlock**: Trajectories rejected by the safety gate can NEVER reach motor drivers; AI agents cannot bypass this check.
- **Safety Checks**:
  1. *Emergency Stop*: Monitored via `EmergencyStopManager`.
  2. *Joint Limits*: Verifies mechanical position bounds and maximum velocities on every trajectory point.
  3. *Workspace Boundaries*: Enforces $(X \in [-0.8, 0.8]\text{m}, Y \in [-0.8, 0.8]\text{m}, Z \in [0.0, 1.2]\text{m})$, preventing table penetration.
  4. *Obstacle Margins*: Enforces $\ge 0.05\text{m}$ clearance from detected obstacles.

---

## 5. Quickstart & Installation

### Step 1: Environment Setup
```bash
# Clone and enter directory
cd VLA-INTELLIGENT-ROBOT

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
```bash
copy .env.example .env
```

### Step 3: Initialize Database (PostgreSQL / In-Memory Fallback)
```bash
python scripts/initialize_database.py
```
*(If PostgreSQL is not running locally, the system automatically uses a thread-safe in-memory SQLite database without failing.)*

### Step 4: Run End-to-End System Demonstration
```bash
python scripts/run_system.py
```

### Step 5: Launch Interactive Web Interface
```bash
python -m uvicorn interface.app:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 6. Running Individual Pipeline Modules

- **Run Computer Vision Pipeline**:
  ```bash
  python scripts/run_vision.py
  ```
- **Run Natural Language Processing**:
  ```bash
  python scripts/run_nlp.py
  ```
- **Train Custom Intent Classifier**:
  ```bash
  python scripts/train_intent_classifier.py
  ```
- **Run Standalone Simulation**:
  ```bash
  python scripts/run_simulation.py
  ```

---

## 7. Running the Test Suite

Run all unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 8. Switching from Simulation to Physical Hardware

Edit `.env` (or configure in `config/settings.py`):
```ini
SIMULATION_MODE=false
ENABLE_ROS2=true
CAMERA_TYPE=realsense_d435i
```

1. Connect Intel RealSense D435i over USB 3.0.
2. Launch ROS 2 robot drivers (`ros2 launch ur_robot_driver ur_control.launch.py`).
3. Launch MoveIt 2 node (`ros2 launch ur_moveit_config ur_moveit.launch.py`).
4. Run `python scripts/run_system.py`. Trajectories will pass through the Independent Safety Gate before executing on physical arm actuators.
