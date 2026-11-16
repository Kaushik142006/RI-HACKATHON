-- PostgreSQL Schema for Vision-Language-Action Intelligent Robot System

-- 1. Conversations Table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 2. Messages Table
CREATE TABLE IF NOT EXISTS messages (
    message_id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(64) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    sender VARCHAR(32) NOT NULL, -- 'user', 'robot', 'system'
    content TEXT NOT NULL,
    audio_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 3. Agent Execution Logs
CREATE TABLE IF NOT EXISTS agent_logs (
    log_id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(64) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    agent_id VARCHAR(64) NOT NULL,
    step_number INT DEFAULT 1,
    status VARCHAR(32) NOT NULL, -- 'started', 'success', 'failed'
    input_state JSONB DEFAULT '{}'::jsonb,
    output_state JSONB DEFAULT '{}'::jsonb,
    execution_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Task History Table
CREATE TABLE IF NOT EXISTS task_history (
    task_id VARCHAR(64) PRIMARY KEY,
    conversation_id VARCHAR(64) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    raw_command TEXT NOT NULL,
    action VARCHAR(64) NOT NULL,
    target_object VARCHAR(64),
    target_color VARCHAR(32),
    destination VARCHAR(64),
    task_status VARCHAR(32) NOT NULL, -- 'pending', 'executing', 'completed', 'failed'
    decomposition JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 5. Execution History Table
CREATE TABLE IF NOT EXISTS execution_history (
    execution_id SERIAL PRIMARY KEY,
    task_id VARCHAR(64) REFERENCES task_history(task_id) ON DELETE CASCADE,
    trajectory_points_count INT DEFAULT 0,
    safety_approved BOOLEAN DEFAULT FALSE,
    safety_telemetry JSONB DEFAULT '{}'::jsonb,
    hardware_status VARCHAR(32) NOT NULL,
    verification_status VARCHAR(32) NOT NULL,
    accuracy_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Error Logs Table
CREATE TABLE IF NOT EXISTS error_logs (
    error_id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(64) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    agent_id VARCHAR(64),
    error_type VARCHAR(64) NOT NULL,
    error_message TEXT NOT NULL,
    recovery_action VARCHAR(64),
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_conv ON agent_logs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_task_hist_conv ON task_history(conversation_id);
