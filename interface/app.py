"""
FastAPI Web Interface for Vision-Language-Action Intelligent Robot System.
Provides interactive human-robot interface supporting voice/text commands,
live pipeline visualization, safety gate status, notifications, and history.
"""

from typing import Dict, Any, Optional
import time
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from orchestration.graph import VLAMultiAgentGraph
from orchestration.state import AgentState
from vision.vision_pipeline import VisionPipeline
from interface.task_monitor import task_monitor
from interface.notifications import NotificationManager
from database.chat_history import ChatHistoryRepository

logger = logging.getLogger(__name__)

app = FastAPI(title="VLA Intelligent Robot System UI", version="1.0.0")

# System instances
graph = VLAMultiAgentGraph(simulation_mode=True)
vision_pipe = VisionPipeline(simulation_mode=True)
chat_repo = ChatHistoryRepository()


class CommandRequest(BaseModel):
    command: str
    conversation_id: Optional[str] = "conv_web_session"
    is_voice: Optional[bool] = False


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLA Intelligent Robot Control System</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .title { font-size: 24px; font-weight: 700; color: #60a5fa; }
        .mode-badge { background: #065f46; color: #34d399; padding: 6px 12px; border-radius: 9999px; font-size: 13px; font-weight: 600; }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
        .card h2 { font-size: 18px; margin-bottom: 16px; color: #93c5fd; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
        
        .input-group { display: flex; gap: 10px; margin-bottom: 14px; }
        input[type="text"] { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); background: #0f172a; color: #fff; font-size: 15px; }
        button { padding: 12px 20px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-dark); }
        .btn-voice { background: #475569; color: white; }
        
        .preset-btns { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
        .btn-preset { background: #334155; color: #cbd5e1; font-size: 12px; padding: 6px 10px; border-radius: 6px; }
        
        .status-box { background: #0f172a; padding: 14px; border-radius: 8px; border-left: 4px solid var(--primary); margin-bottom: 12px; font-size: 14px; }
        .safety-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 12px; margin-top: 8px; }
        .badge-safe { background: #065f46; color: #34d399; }
        .badge-danger { background: #7f1d1d; color: #f87171; }
        
        .pipeline { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
        .stage { display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 10px 14px; border-radius: 6px; font-size: 13px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .dot-green { background: var(--success); }
        .dot-gray { background: #64748b; }
        
        .terminal { background: #000; color: #10b981; font-family: monospace; padding: 14px; border-radius: 8px; height: 160px; overflow-y: auto; font-size: 12px; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">Vision-Language-Action Intelligent Robot System</div>
            <div style="font-size: 13px; color: var(--text-dim); margin-top: 4px;">16-Agent LangGraph Multi-Agent Architecture with Independent Safety Gate</div>
        </div>
        <div class="mode-badge">MODE: SIMULATION ACTIVE</div>
    </div>

    <div class="grid">
        <div>
            <!-- Command Input Card -->
            <div class="card">
                <h2>Human Command Input</h2>
                <div class="preset-btns">
                    <span style="font-size: 12px; color: var(--text-dim); margin-right: 6px;">Presets:</span>
                    <button class="btn-preset" onclick="setCmd('Pick up the red bottle and place it inside the storage box.')">Benchmark Pick & Place</button>
                    <button class="btn-preset" onclick="setCmd('Grab the blue cup and put it on the tray.')">Move Blue Cup</button>
                    <button class="btn-preset" onclick="setCmd('Inspect the workspace.')">Inspect Environment</button>
                </div>
                <div class="input-group">
                    <input type="text" id="cmdInput" placeholder="Enter voice or text command (e.g. 'Pick up the red bottle...')" value="Pick up the red bottle and place it inside the storage box.">
                    <button class="btn-voice" onclick="simulateVoice()">🎤 Voice</button>
                    <button class="btn-primary" onclick="submitCommand()">Execute</button>
                </div>
                <div id="notifArea"></div>
            </div>

            <!-- Task Status Card -->
            <div class="card">
                <h2>Live Task Status & Safety Gate</h2>
                <div class="status-box" id="taskStatusBox">
                    <strong>Status:</strong> Ready for execution.<br>
                    <div class="safety-badge badge-safe">INDEPENDENT SAFETY GATE: ARMED & ACTIVE</div>
                </div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 10px;" id="verificationDetails"></div>
            </div>
        </div>

        <div>
            <!-- Pipeline Progress -->
            <div class="card">
                <h2>16-Agent Pipeline Execution Flow</h2>
                <div class="pipeline" id="pipelineFlow">
                    <div class="stage"><span>1. Voice & NLP Preprocessing (Whisper, NLTK, Intent)</span><span class="dot dot-gray" id="dot1"></span></div>
                    <div class="stage"><span>2. Vision Perception (RealSense, YOLOv8, 3D Coordinates)</span><span class="dot dot-gray" id="dot2"></span></div>
                    <div class="stage"><span>3. Workspace Environment Check (Clutter & Hazards)</span><span class="dot dot-gray" id="dot3"></span></div>
                    <div class="stage"><span>4. Work Splitter & Task Decomposition (7 Primitives)</span><span class="dot dot-gray" id="dot4"></span></div>
                    <div class="stage"><span>5. MoveIt 2 Motion Planning (Inverse Kinematics)</span><span class="dot dot-gray" id="dot5"></span></div>
                    <div class="stage"><span>6. Independent Safety Gate (Non-AI Trajectory Audit)</span><span class="dot dot-gray" id="dot6"></span></div>
                    <div class="stage"><span>7. ros2_control Hardware & Verification</span><span class="dot dot-gray" id="dot7"></span></div>
                </div>
            </div>

            <!-- Live Logs Terminal -->
            <div class="card">
                <h2>System Execution Telemetry</h2>
                <div class="terminal" id="terminalLog">
                    [SYSTEM READY] Listening for human commands on port 8000...<br>
                    [CONFIG] Defaulting to Simulation-First Mode (Safe Hardware Interlock).
                </div>
            </div>
        </div>
    </div>

    <script>
        function setCmd(text) {
            document.getElementById('cmdInput').value = text;
        }

        function simulateVoice() {
            document.getElementById('cmdInput').value = "Pick up the red bottle and place it inside the storage box.";
            submitCommand(true);
        }

        async function submitCommand(isVoice = false) {
            const cmd = document.getElementById('cmdInput').value;
            const term = document.getElementById('terminalLog');
            term.innerHTML += `<br>[USER] Executing: "${cmd}" (Voice: ${isVoice})`;

            // Light up stage 1
            document.getElementById('dot1').className = 'dot dot-green';

            try {
                const response = await fetch('/api/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: cmd, is_voice: isVoice })
                });
                const data = await response.json();
                
                // Light up remaining dots
                for (let i = 2; i <= 7; i++) {
                    document.getElementById('dot' + i).className = 'dot dot-green';
                }

                document.getElementById('taskStatusBox').innerHTML = `
                    <strong>Command:</strong> ${data.command}<br>
                    <strong>Result:</strong> ${data.final_response}<br>
                    <div class="safety-badge ${data.safety_approval ? 'badge-safe' : 'badge-danger'}">
                        INDEPENDENT SAFETY GATE: ${data.safety_approval ? 'TRAJECTORY APPROVED' : 'REJECTED'}
                    </div>
                `;

                if (data.evaluation) {
                    document.getElementById('verificationDetails').innerHTML = `
                        <strong>Evaluation Accuracy:</strong> ${data.evaluation.accuracy_score_percent}% | 
                        <strong>Placement Deviation:</strong> ${data.evaluation.placement_error_m}m | 
                        <strong>Primitives Executed:</strong> ${data.evaluation.total_planning_steps}
                    `;
                }

                term.innerHTML += `<br>[ORCHESTRATOR] Success: ${data.final_response}`;
                term.scrollTop = term.scrollHeight;
            } catch (err) {
                term.innerHTML += `<br>[ERROR] Execution failed: ${err}`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve single-page web dashboard."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/command")
async def execute_command(req: CommandRequest):
    """Process natural language or voice command through full 16-agent graph."""
    logger.info(f"Received web command: '{req.command}'")

    # 1. Capture current vision state
    vision_state = vision_pipe.process_frame()

    # 2. Build initial AgentState
    state = AgentState(
        conversation_id=req.conversation_id or "conv_web",
        original_voice_command=req.command,
        transcribed_text=req.command,
        vision_state=vision_state.model_dump()
    )

    # 3. Execute Multi-Agent Graph
    final_state = graph.execute(state)

    return JSONResponse({
        "status": "success",
        "command": req.command,
        "target_object": final_state.requested_object,
        "destination": final_state.requested_destination,
        "safety_approval": final_state.safety_approval,
        "robot_status": final_state.robot_execution_status,
        "verification": final_state.verification_results,
        "evaluation": final_state.evaluation_metrics,
        "final_response": final_state.final_response
    })


@app.get("/api/status")
async def get_system_status():
    """Return task monitor status."""
    return task_monitor.get_status()
