use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::{Obfuscator, StreamConfig};
use std::io::Cursor;

#[tokio::test]
async fn test_stream_file_different_chunk_sizes() {
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
    ];

    let test_data = "Line 1 with email test@example.com and ssn 123-45-6789\n\
                     Line 2 with email user@domain.org and ssn 987-65-4321\n\
                     Line 3 with email admin@company.net and ssn 555-12-3456\n\
                     Line 4 with more data and another email contact@business.com\n";

    let chunk_sizes = vec![8, 32, 128, 512]; // Small sizes to test line handling

    for chunk_size in chunk_sizes {
        let mut obfuscator = Obfuscator::new(&rules).unwrap();
        let reader = Cursor::new(test_data.as_bytes());
        let mut output = Vec::new();
        let config = StreamConfig { chunk_size };

        obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
        
        let result = String::from_utf8(output).unwrap();
        
        // Verify emails and SSNs are obfuscated
        assert!(!result.contains("test@example.com"));
        assert!(!result.contains("123-45-6789"));
        assert!(result.contains("[EMAIL-"));
        assert!(result.contains("[SSN-"));
        
        // Verify all lines are present
        assert_eq!(result.lines().count(), 4);
    }
}

#[tokio::test] 
async fn test_stream_file_large_lines() {
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
    ];

    // Create a line larger than typical chunk sizes
    let large_line = format!("Very long line with email test@example.com and lots of data: {}\n", 
                            "x".repeat(1000));
    
    let mut obfuscator = Obfuscator::new(&rules).unwrap();
    let reader = Cursor::new(large_line.as_bytes());
    let mut output = Vec::new();
    let config = StreamConfig { chunk_size: 64 }; // Smaller than line

    obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
    
    let result = String::from_utf8(output).unwrap();
    assert!(!result.contains("test@example.com"));
    assert!(result.contains("[EMAIL-0]"));
}

#[tokio::test]
async fn test_stream_file_performance_correctness() {
    // Verify that different chunk sizes produce identical results
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
    ];

    let test_data = (0..100)
        .map(|i| format!("Line {} with email user{}@test.com\n", i, i))
        .collect::<String>();

    let chunk_sizes = vec![8 * 1024, 256 * 1024, 1024 * 1024];
    let mut results = Vec::new();

    for chunk_size in chunk_sizes {
        let mut obfuscator = Obfuscator::new(&rules).unwrap();
        let reader = Cursor::new(test_data.as_bytes());
        let mut output = Vec::new();
        let config = StreamConfig { chunk_size };

        obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
        results.push(String::from_utf8(output).unwrap());
    }

    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], 
                  "Results differ between chunk sizes");
    }
}