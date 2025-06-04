use data_obfuscator::errors::AppError;
use data_obfuscator::obfuscator::ObfuscationError;
use data_obfuscator::llm_client::LlmError;

#[test]
fn app_error_from_obfuscation_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, "fail");
    let app: AppError = ObfuscationError::Io(io_err).into();
    assert!(matches!(app, AppError::Obfuscation(ObfuscationError::Io(_))));
}

#[test]
fn app_error_from_llm_invalid_response() {
    let app: AppError = LlmError::InvalidResponse.into();
    assert!(matches!(app, AppError::Llm(LlmError::InvalidResponse)));
}
