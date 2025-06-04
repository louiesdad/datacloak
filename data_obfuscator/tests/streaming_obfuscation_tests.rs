use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::Obfuscator;
use tokio::io::{BufReader, BufWriter};

#[tokio::test]
async fn obfuscates_stream() {
    let rules = vec![Rule { pattern: "\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b".into(), label: "EMAIL".into() }];
    let mut obfuscator = Obfuscator::new(&rules).unwrap();
    let input = "Contact: test@example.com\nSend mail.";
    let reader = BufReader::new(input.as_bytes());
    let mut output: Vec<u8> = Vec::new();
    {
        let writer = BufWriter::new(&mut output);
        obfuscator.obfuscate_stream(reader, writer).await.unwrap();
    }
    let out_str = String::from_utf8(output).unwrap();
    assert!(out_str.contains("[EMAIL-0]"));
}
