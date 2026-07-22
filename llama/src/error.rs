use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LlamaError {
    #[error("llama: failed to load model")]
    ModelLoad,
    #[error("llama: failed to load state")]
    StateLoad,
    #[error("llama: inference failed")]
    Inference,
    #[error("llama: out of memory allocating result buffer")]
    OutOfMemory,
    #[error("llama: model loaded without embeddings")]
    EmbeddingsDisabled,
    #[error("llama: not implemented")]
    NotImplemented,
}

pub type Result<T> = std::result::Result<T, LlamaError>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn messages_match_go() {
        assert_eq!(LlamaError::ModelLoad.to_string(), "llama: failed to load model");
        assert_eq!(LlamaError::EmbeddingsDisabled.to_string(), "llama: model loaded without embeddings");
        assert_eq!(LlamaError::NotImplemented.to_string(), "llama: not implemented");
    }
}
