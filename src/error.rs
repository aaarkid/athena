use std::fmt;

/// Result type for Athena operations
pub type Result<T> = std::result::Result<T, AthenaError>;

/// Main error type for Athena library
#[derive(Debug, Clone)]
pub enum AthenaError {
    /// Invalid dimensions for operations
    DimensionMismatch {
        expected: String,
        actual: String,
    },
    
    /// Invalid parameter value
    InvalidParameter {
        name: String,
        reason: String,
    },
    
    /// IO errors (file operations)
    IoError(String),
    
    /// Serialization/deserialization errors
    SerializationError(String),
    
    /// Numerical computation errors
    NumericalError(String),
    
    /// Empty buffer or container
    EmptyBuffer(String),
    
    /// Invalid action
    InvalidAction {
        action: usize,
        max_actions: usize,
    },
    
    /// Training error
    TrainingError(String),
}

impl fmt::Display for AthenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AthenaError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            AthenaError::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{}': {}", name, reason)
            }
            AthenaError::IoError(msg) => write!(f, "IO error: {}", msg),
            AthenaError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            AthenaError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            AthenaError::EmptyBuffer(msg) => write!(f, "Empty buffer: {}", msg),
            AthenaError::InvalidAction { action, max_actions } => {
                write!(f, "Invalid action {}: must be less than {}", action, max_actions)
            }
            AthenaError::TrainingError(msg) => write!(f, "Training error: {}", msg),
        }
    }
}

impl std::error::Error for AthenaError {}

// Conversion from std::io::Error
impl From<std::io::Error> for AthenaError {
    fn from(err: std::io::Error) -> Self {
        AthenaError::IoError(err.to_string())
    }
}

// Conversion from bincode::Error
impl From<bincode::Error> for AthenaError {
    fn from(err: bincode::Error) -> Self {
        AthenaError::SerializationError(err.to_string())
    }
}

// Helper functions for common error patterns
impl AthenaError {
    pub fn dimension_mismatch<S: Into<String>>(expected: S, actual: S) -> Self {
        AthenaError::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
    
    pub fn invalid_parameter<S: Into<String>>(name: S, reason: S) -> Self {
        AthenaError::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }
}