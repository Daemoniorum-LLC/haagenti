//! Test reference decompression across sizes

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn test_pattern(name: &str, data: &[u8]) {
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let compressed = ctx.compress(data).unwrap();

    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(dec) => {
            if dec == data {
                println!(
                    "{}: PASS ({} -> {} bytes)",
                    name,
                    data.len(),
                    compressed.len()
                );
            } else {
                println!(
                    "{}: MISMATCH ({} -> {} bytes)",
                    name,
                    data.len(),
                    compressed.len()
                );
            }
        }
        Err(e) => {
            println!(
                "{}: FAIL ({} -> {} bytes): {}",
                name,
                data.len(),
                compressed.len(),
                e
            );
        }
    }
}

fn main() {
    println!("=== Testing Reference Interop ===\n");

    // Uniform data
    test_pattern("A*100", &[b'A'; 100]);
    test_pattern("A*1000", &vec![b'A'; 1000]);
    test_pattern("A*10000", &vec![b'A'; 10000]);
    test_pattern("A*65536", &vec![b'A'; 65536]);

    println!();

    // Multi-phase at different sizes
    for size in [100, 200, 500, 1000, 4000, 8000, 16000, 32000, 65536] {
        let mut data = Vec::new();
        let chunk = size / 4;
        data.extend(vec![b'X'; chunk]);
        data.extend(vec![b'Y'; chunk]);
        data.extend(vec![b'Z'; chunk]);
        data.extend(vec![b'W'; size - 3 * chunk]);
        test_pattern(&format!("XYZW*{}", size), &data);
    }

    println!();

    // Text patterns
    let text = b"The quick brown fox jumps over the lazy dog. ";
    for size in [100, 500, 1000, 4000, 8000, 16000, 65536] {
        let data: Vec<u8> = text.iter().cycle().take(size).copied().collect();
        test_pattern(&format!("text*{}", size), &data);
    }
}
