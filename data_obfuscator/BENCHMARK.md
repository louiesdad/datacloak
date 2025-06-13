# Streaming Performance Benchmark Results

This document presents the performance analysis of different chunk sizes for streaming file obfuscation.

## Benchmark Setup

- **Test Data**: 10MB file with email and SSN patterns for obfuscation
- **Patterns**: Email regex (`\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b`) and SSN regex (`\b\d{3}-\d{2}-\d{4}\b`)
- **Environment**: Rust release build with optimizations
- **Measurement**: 10 samples per chunk size, 10-second measurement window

## Performance Results

### Streaming Performance by Chunk Size (10MB file)

| Chunk Size | Mean Time | Performance | Memory Usage | Throughput |
|------------|-----------|-------------|--------------|------------|
| 8KB        | 77.06 ms  | Baseline    | Low          | ~130 MB/s  |
| 256KB      | 74.68 ms  | +3.1%       | Medium       | ~134 MB/s  |
| 1MB        | 74.41 ms  | **+3.6%**   | High         | ~134 MB/s  |

### Analysis

#### Performance Characteristics

1. **Optimal Performance**: 1MB chunk size provides the best performance with 74.41ms mean time
2. **Diminishing Returns**: Beyond 256KB, performance gains level off
3. **Memory Trade-off**: Larger chunks use more memory but provide better throughput

#### Key Findings

- **8KB chunks**: Suitable for low-memory environments, ~3% slower
- **256KB chunks** (default): Good balance of performance and memory usage
- **1MB chunks**: Best performance but higher memory usage

#### Resource Usage

| Chunk Size | Memory per Chunk | CPU Cache Efficiency | I/O Patterns |
|------------|------------------|---------------------|--------------|
| 8KB        | 8,192 bytes      | High                | Many small reads |
| 256KB      | 262,144 bytes    | Good                | Balanced |
| 1MB        | 1,048,576 bytes  | Lower               | Few large reads |

## Recommendations

### Production Usage

1. **Default Configuration**: Use 256KB chunks for balanced performance
2. **High-Performance**: Use 1MB chunks when memory is abundant
3. **Memory-Constrained**: Use 8KB chunks in memory-limited environments

### Configuration Examples

```bash
# Default (256KB)
cargo run --document-path large_file.txt

# High performance (1MB)
cargo run --document-path large_file.txt --chunk-size 1048576

# Memory efficient (8KB)  
cargo run --document-path large_file.txt --chunk-size 8192
```

## Implementation Details

### StreamConfig Structure

```rust
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub chunk_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024, // 256KB default
        }
    }
}
```

### Usage in Code

```rust
let config = StreamConfig { chunk_size: 1024 * 1024 }; // 1MB
obfuscator.stream_file(reader, writer, &config).await?;
```

## Scalability Analysis

### Large File Performance (Extrapolated)

| File Size | 8KB Chunks | 256KB Chunks | 1MB Chunks |
|-----------|------------|--------------|------------|
| 100MB     | ~771ms     | ~747ms       | ~744ms     |
| 1GB       | ~7.7s      | ~7.5s        | ~7.4s      |
| 10GB      | ~77s       | ~75s         | ~74s       |

### Throughput Comparison

- **8KB**: ~130 MB/s
- **256KB**: ~134 MB/s  
- **1MB**: ~134 MB/s

## Testing

### Verification Tests

All chunk sizes produce identical obfuscation results, ensuring correctness across configurations:

```bash
cargo test --test streaming_chunk_test
```

### Benchmark Execution

```bash
# Quick benchmark (10MB)
cargo bench --bench quick_streaming_benchmark

# Full benchmark suite  
cargo bench --bench streaming_benchmark
```

## Conclusion

The streaming optimization successfully provides:

1. **Configurable Performance**: 3-4% improvement with larger chunks
2. **Memory Flexibility**: Support for memory-constrained to high-performance environments  
3. **Consistent Results**: All chunk sizes produce identical obfuscation output
4. **Production Ready**: Default 256KB provides optimal balance for most use cases

The implementation enables processing files of any size with predictable memory usage and excellent throughput characteristics.