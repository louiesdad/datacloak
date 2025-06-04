use std::collections::HashMap;
use data_obfuscator::deobfuscator::deobfuscate_text;

#[test]
fn restores_text() {
    let mut map = HashMap::new();
    map.insert("[EMAIL-0]".to_string(), "test@example.com".to_string());
    let result = deobfuscate_text("hello [EMAIL-0]", &map);
    assert_eq!(result, "hello test@example.com");
}
