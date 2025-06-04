use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::Obfuscator;
use tokio::io::BufReader;

#[tokio::test]
async fn stream_obfuscates_multiple_lines() {
    let rules = vec![Rule {
        pattern: "\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b".into(),
        label: "EMAIL".into(),
    }];
    let mut obfuscator = Obfuscator::new(&rules).unwrap();

    let data = "first test@example.com\nsecond john@doe.com\n";
    let reader = BufReader::new(data.as_bytes());
    let mut out = Vec::new();

    obfuscator.obfuscate_stream(reader, &mut out).await.unwrap();

    let out_str = String::from_utf8(out).unwrap();
    assert!(out_str.contains("[EMAIL-0]"));
    assert!(out_str.contains("[EMAIL-1]"));

    let map = obfuscator.placeholder_map();
    let de = data_obfuscator::deobfuscator::deobfuscate_text(&out_str, map);
    assert_eq!(de, data);
}

#[tokio::test]
async fn stream_reuses_tokens_across_lines() {
    let rules = vec![Rule {
        pattern: "\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b".into(),
        label: "EMAIL".into(),
    }];
    let mut obfuscator = Obfuscator::new(&rules).unwrap();

    let data = "repeat test@example.com\nagain test@example.com\n";
    let reader = BufReader::new(data.as_bytes());
    let mut out = Vec::new();

    obfuscator.obfuscate_stream(reader, &mut out).await.unwrap();

    let out_str = String::from_utf8(out).unwrap();

    // both occurrences should use the same placeholder
    assert_eq!(out_str.matches("[EMAIL-0]").count(), 2);

    let map = obfuscator.placeholder_map();
    let de = data_obfuscator::deobfuscator::deobfuscate_text(&out_str, map);
    assert_eq!(de, data);
}

