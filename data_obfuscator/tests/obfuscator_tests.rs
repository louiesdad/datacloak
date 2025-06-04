use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::Obfuscator;

#[test]
fn obfuscates_email() {
    let rules = vec![Rule { pattern: "\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b".into(), label: "EMAIL".into() }];
    let mut obfuscator = Obfuscator::new(&rules).unwrap();
    let input = "Contact me at test@example.com";
    let out = obfuscator.obfuscate_text(input);
    assert!(out.contains("[EMAIL-0]"));
    let map = obfuscator.placeholder_map();
    let de = data_obfuscator::deobfuscator::deobfuscate_text(&out, map);
    assert_eq!(de, input);
}
