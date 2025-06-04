use data_obfuscator::llm_client::LlmClient;

#[tokio::test]
async fn echo_chat() {
    let client = LlmClient::new("http://localhost".to_string(), "key".to_string());
    let reply = client.chat("hello").await.unwrap();
    assert_eq!(reply, "echo: hello");
}
