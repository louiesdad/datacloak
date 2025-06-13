use data_obfuscator::llm_client::LlmClient;
use mockito::Server;

#[tokio::test]
async fn echo_chat() {
    let mut server = Server::new_async().await;
    
    let _m = server.mock("POST", "/")
        .match_header("authorization", "Bearer key")
        .match_header("content-type", "application/json")
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "echo: hello" } } ] }"#)
        .create_async()
        .await;

    let client = LlmClient::new(server.url(), "key".to_string());
    let reply = client.chat("hello").await.unwrap();
    assert_eq!(reply, "echo: hello");
}

#[tokio::test]
async fn chat_returns_content() {
    let mut server = Server::new_async().await;
    
    let _m = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", "application/json")
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "hello" } } ] }"#)
        .create_async()
        .await;

    let client = LlmClient::new(server.url(), "test-key".into());
    let resp = client.chat("ping").await.unwrap();
    assert_eq!(resp, "hello");
}