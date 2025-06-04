use data_obfuscator::llm_client::LlmClient;
use mockito::{mock, Matcher};

#[tokio::test]
async fn echo_chat() {
    let client = LlmClient::new("http://localhost".to_string(), "key".to_string());
    let reply = client.chat("hello").await.unwrap();
    assert_eq!(reply, "echo: hello");
}

#[tokio::test]
async fn chat_returns_content() {
    let _m = mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", Matcher::Regex("application/json".into()))
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "hello" } } ] }"#)
        .create();

    let client = LlmClient::new(mockito::server_url(), "test-key".into());
    let resp = client.chat("ping").await.unwrap();
    assert_eq!(resp, "hello");
}