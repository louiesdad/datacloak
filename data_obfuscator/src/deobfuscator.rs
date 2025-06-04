use std::collections::HashMap;

pub fn deobfuscate_text(input: &str, map: &HashMap<String, String>) -> String {
    let mut output = input.to_string();
    for (token, original) in map {
        output = output.replace(token, original);
    }
    output
}
