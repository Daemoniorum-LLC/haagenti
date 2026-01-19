//! Debug the RLE-like pattern issue - isolate the problem

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn test_pattern(name: &str, data: &[u8]) {
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let compressed = ctx.compress(data).unwrap();

    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(decompressed) => {
            if decompressed == data {
                println!("{}: PASS ({} -> {} bytes, {:.1}x)", name, data.len(), compressed.len(),
                         data.len() as f64 / compressed.len() as f64);
            } else {
                println!("{}: MISMATCH ({} -> {} bytes)", name, data.len(), compressed.len());
            }
        }
        Err(e) => {
            println!("{}: FAIL ({} -> {} bytes): {}", name, data.len(), compressed.len(), e);
        }
    }
}

fn main() {
    // Test various patterns to isolate the issue
    println!("=== Pattern Tests ===\n");

    // Simple repeats
    test_pattern("A*100", &vec![b'A'; 100]);
    test_pattern("A*200", &vec![b'A'; 200]);
    test_pattern("A*400", &vec![b'A'; 400]);

    // Two-phase patterns
    let mut xy = Vec::new();
    xy.extend(vec![b'X'; 100]);
    xy.extend(vec![b'Y'; 100]);
    test_pattern("X*100 Y*100", &xy);

    // Three-phase patterns
    let mut xyz = Vec::new();
    xyz.extend(vec![b'X'; 100]);
    xyz.extend(vec![b'Y'; 100]);
    xyz.extend(vec![b'Z'; 100]);
    test_pattern("X*100 Y*100 Z*100", &xyz);

    // Four-phase patterns (the failing case)
    let mut xyzx = Vec::new();
    xyzx.extend(vec![b'X'; 100]);
    xyzx.extend(vec![b'Y'; 100]);
    xyzx.extend(vec![b'Z'; 100]);
    xyzx.extend(vec![b'X'; 100]);
    test_pattern("X*100 Y*100 Z*100 X*100", &xyzx);

    // Variations to find the boundary
    let mut xyxy = Vec::new();
    xyxy.extend(vec![b'X'; 100]);
    xyxy.extend(vec![b'Y'; 100]);
    xyxy.extend(vec![b'X'; 100]);
    xyxy.extend(vec![b'Y'; 100]);
    test_pattern("X*100 Y*100 X*100 Y*100", &xyxy);

    // Check if it's the repeated X causing issues
    let mut xyx = Vec::new();
    xyx.extend(vec![b'X'; 100]);
    xyx.extend(vec![b'Y'; 100]);
    xyx.extend(vec![b'X'; 100]);
    test_pattern("X*100 Y*100 X*100", &xyx);

    // Check smaller sizes
    let mut xyzx_small = Vec::new();
    xyzx_small.extend(vec![b'X'; 50]);
    xyzx_small.extend(vec![b'Y'; 50]);
    xyzx_small.extend(vec![b'Z'; 50]);
    xyzx_small.extend(vec![b'X'; 50]);
    test_pattern("X*50 Y*50 Z*50 X*50", &xyzx_small);

    // Reference zstd for comparison
    println!("\n=== Reference zstd ===\n");
    let data = xyzx.clone();
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();
    println!("Reference: {} -> {} bytes", data.len(), ref_compressed.len());
}
