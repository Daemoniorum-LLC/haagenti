//! Stream control and mid-generation commands

use crate::{Result, StreamError, StreamState};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

/// Commands that can be sent to a running stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    /// Pause generation
    Pause,
    /// Resume generation
    Resume,
    /// Cancel generation
    Cancel,
    /// Redirect to new prompt (if supported)
    Redirect { new_prompt: String },
    /// Adjust step count
    AdjustSteps { new_total: u32 },
    /// Change preview quality
    SetQuality { quality: crate::PreviewQuality },
    /// Request immediate preview
    RequestPreview,
    /// Skip to final
    SkipToFinal,
    /// Get current status
    GetStatus,
}

/// Response to a control command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlResponse {
    /// Command acknowledged
    Ok,
    /// Command rejected with reason
    Rejected { reason: String },
    /// Status response
    Status {
        state: StreamState,
        step: u32,
        total_steps: u32,
        progress: f32,
    },
    /// Error response
    Error { message: String },
}

/// Controller for managing stream commands
#[derive(Debug)]
pub struct StreamController {
    /// Command sender
    command_tx: mpsc::Sender<(ControlCommand, oneshot::Sender<ControlResponse>)>,
    /// Command receiver
    command_rx: Option<mpsc::Receiver<(ControlCommand, oneshot::Sender<ControlResponse>)>>,
    /// Current state
    state: std::sync::Arc<std::sync::RwLock<StreamState>>,
}

impl StreamController {
    /// Create a new controller
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(16);
        Self {
            command_tx: tx,
            command_rx: Some(rx),
            state: std::sync::Arc::new(std::sync::RwLock::new(StreamState::Idle)),
        }
    }

    /// Send a command and wait for response
    pub async fn send(&self, command: ControlCommand) -> Result<ControlResponse> {
        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx
            .send((command, response_tx))
            .await
            .map_err(|_| StreamError::ChannelError("Command channel closed".into()))?;

        response_rx
            .await
            .map_err(|_| StreamError::ChannelError("Response channel closed".into()))
    }

    /// Send command without waiting for response
    pub async fn send_fire_and_forget(&self, command: ControlCommand) -> Result<()> {
        let (response_tx, _response_rx) = oneshot::channel();

        self.command_tx
            .send((command, response_tx))
            .await
            .map_err(|_| StreamError::ChannelError("Command channel closed".into()))?;

        Ok(())
    }

    /// Take the command receiver (for the stream handler)
    pub fn take_receiver(
        &mut self,
    ) -> Option<mpsc::Receiver<(ControlCommand, oneshot::Sender<ControlResponse>)>> {
        self.command_rx.take()
    }

    /// Convenience: pause
    pub async fn pause(&self) -> Result<ControlResponse> {
        self.send(ControlCommand::Pause).await
    }

    /// Convenience: resume
    pub async fn resume(&self) -> Result<ControlResponse> {
        self.send(ControlCommand::Resume).await
    }

    /// Convenience: cancel
    pub async fn cancel(&self) -> Result<ControlResponse> {
        self.send(ControlCommand::Cancel).await
    }

    /// Convenience: get status
    pub async fn status(&self) -> Result<ControlResponse> {
        self.send(ControlCommand::GetStatus).await
    }

    /// Convenience: request preview
    pub async fn request_preview(&self) -> Result<ControlResponse> {
        self.send(ControlCommand::RequestPreview).await
    }

    /// Get cached state
    pub fn cached_state(&self) -> StreamState {
        *self.state.read().unwrap()
    }

    /// Update cached state
    pub fn update_state(&self, state: StreamState) {
        *self.state.write().unwrap() = state;
    }
}

impl Default for StreamController {
    fn default() -> Self {
        Self::new()
    }
}

/// Command handler that processes commands from controller
pub struct CommandHandler {
    /// Receiver for commands
    receiver: mpsc::Receiver<(ControlCommand, oneshot::Sender<ControlResponse>)>,
}

impl CommandHandler {
    /// Create from controller
    pub fn from_controller(controller: &mut StreamController) -> Option<Self> {
        controller.take_receiver().map(|rx| Self { receiver: rx })
    }

    /// Receive next command
    pub async fn recv(&mut self) -> Option<(ControlCommand, oneshot::Sender<ControlResponse>)> {
        self.receiver.recv().await
    }

    /// Try to receive command without blocking
    pub fn try_recv(&mut self) -> Option<(ControlCommand, oneshot::Sender<ControlResponse>)> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_controller_send() {
        let mut controller = StreamController::new();
        let mut handler = CommandHandler::from_controller(&mut controller).unwrap();

        // Spawn a handler task
        let handle = tokio::spawn(async move {
            if let Some((cmd, responder)) = handler.recv().await {
                match cmd {
                    ControlCommand::GetStatus => {
                        let _ = responder.send(ControlResponse::Status {
                            state: StreamState::Running,
                            step: 5,
                            total_steps: 20,
                            progress: 25.0,
                        });
                    }
                    _ => {
                        let _ = responder.send(ControlResponse::Ok);
                    }
                }
            }
        });

        let response = controller.status().await.unwrap();
        match response {
            ControlResponse::Status { step, .. } => {
                assert_eq!(step, 5);
            }
            _ => panic!("Expected status response"),
        }

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_fire_and_forget() {
        let mut controller = StreamController::new();
        let _handler = CommandHandler::from_controller(&mut controller).unwrap();

        // This should not block even if no one is receiving
        let result = controller.send_fire_and_forget(ControlCommand::Pause).await;
        assert!(result.is_ok());
    }
}
