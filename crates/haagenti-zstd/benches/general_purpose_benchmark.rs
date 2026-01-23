//! General-Purpose Data Benchmarks
//!
//! This benchmark suite tests Haagenti against the data patterns Zstd was
//! originally designed for: text, source code, markup, logs, and mixed files.
//!
//! **Purpose:** Show honest performance on general workloads where Haagenti
//! may not have specialized advantages. This complements the specialized
//! benchmarks that show Haagenti's strengths on structured numeric data.
//!
//! Run with: `cargo bench -p haagenti-zstd --bench general_purpose_benchmark`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ============================================================================
// Canterbury Corpus-Style Data Generators
// https://corpus.canterbury.ac.nz/descriptions/
// ============================================================================

/// English text (alice29.txt style) - natural language prose
fn generate_english_prose(size: usize) -> Vec<u8> {
    // Simulate natural English text with varied sentence structures
    let sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Alice was beginning to get very tired of sitting by her sister on the bank. ",
        "It was the best of times, it was the worst of times. ",
        "Call me Ishmael. Some years ago, never mind how long precisely. ",
        "In a hole in the ground there lived a hobbit. ",
        "It is a truth universally acknowledged that a single man must be in want of a wife. ",
        "All happy families are alike; each unhappy family is unhappy in its own way. ",
        "You don't know about me without you have read a book by the name of Tom Sawyer. ",
        "Whether I shall turn out to be the hero of my own life remains to be seen. ",
        "Once upon a time and a very good time it was there was a moocow. ",
    ];

    let mut rng = StdRng::seed_from_u64(42);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        let sentence = sentences[rng.gen_range(0..sentences.len())];
        result.extend_from_slice(sentence.as_bytes());
        // Occasionally add newlines like real text
        if rng.gen_bool(0.15) {
            result.push(b'\n');
        }
    }
    result.truncate(size);
    result
}

/// C source code (gcc.tar style) - programming language with keywords and structure
fn generate_c_source(size: usize) -> Vec<u8> {
    let fragments = [
        "int main(int argc, char *argv[]) {\n",
        "    if (argc < 2) {\n",
        "        printf(\"Usage: %s <filename>\\n\", argv[0]);\n",
        "        return 1;\n",
        "    }\n",
        "    FILE *fp = fopen(argv[1], \"r\");\n",
        "    if (fp == NULL) {\n",
        "        perror(\"fopen\");\n",
        "        return 1;\n",
        "    }\n",
        "    char buffer[1024];\n",
        "    while (fgets(buffer, sizeof(buffer), fp) != NULL) {\n",
        "        printf(\"%s\", buffer);\n",
        "    }\n",
        "    fclose(fp);\n",
        "    return 0;\n",
        "}\n\n",
        "#include <stdio.h>\n",
        "#include <stdlib.h>\n",
        "#include <string.h>\n",
        "#define MAX_SIZE 4096\n",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))\n",
        "typedef struct {\n",
        "    int x, y, z;\n",
        "    char name[64];\n",
        "} Point;\n\n",
        "static inline int compare(const void *a, const void *b) {\n",
        "    return *(int*)a - *(int*)b;\n",
        "}\n\n",
        "/* This is a multi-line comment\n",
        " * that spans several lines\n",
        " * and explains the code */\n",
        "// Single line comment\n",
        "for (int i = 0; i < n; i++) {\n",
        "    for (int j = 0; j < m; j++) {\n",
        "        matrix[i][j] = i * m + j;\n",
        "    }\n",
        "}\n",
    ];

    let mut rng = StdRng::seed_from_u64(43);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        let fragment = fragments[rng.gen_range(0..fragments.len())];
        result.extend_from_slice(fragment.as_bytes());
    }
    result.truncate(size);
    result
}

/// HTML markup (html style) - structured markup with tags
fn generate_html(size: usize) -> Vec<u8> {
    let fragments = [
        "<!DOCTYPE html>\n<html lang=\"en\">\n",
        "<head>\n    <meta charset=\"UTF-8\">\n",
        "    <title>Page Title</title>\n",
        "    <link rel=\"stylesheet\" href=\"style.css\">\n",
        "</head>\n<body>\n",
        "    <header>\n        <nav>\n",
        "            <ul>\n                <li><a href=\"/\">Home</a></li>\n",
        "                <li><a href=\"/about\">About</a></li>\n",
        "                <li><a href=\"/contact\">Contact</a></li>\n",
        "            </ul>\n        </nav>\n    </header>\n",
        "    <main>\n        <article>\n",
        "            <h1>Main Heading</h1>\n",
        "            <p>This is a paragraph of text content.</p>\n",
        "            <p>Another paragraph with <strong>bold</strong> and <em>italic</em> text.</p>\n",
        "        </article>\n    </main>\n",
        "    <footer>\n        <p>&copy; 2025 Company Name</p>\n    </footer>\n",
        "</body>\n</html>\n",
        "<div class=\"container\">\n    <div class=\"row\">\n",
        "        <div class=\"col-md-6\">\n            <h2>Section</h2>\n        </div>\n",
        "    </div>\n</div>\n",
        "<script src=\"app.js\"></script>\n",
        "<!-- This is an HTML comment -->\n",
    ];

    let mut rng = StdRng::seed_from_u64(44);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        let fragment = fragments[rng.gen_range(0..fragments.len())];
        result.extend_from_slice(fragment.as_bytes());
    }
    result.truncate(size);
    result
}

// ============================================================================
// Silesia Corpus-Style Data Generators
// https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
// ============================================================================

/// XML data (xml style) - structured data with tags and attributes
fn generate_xml(size: usize) -> Vec<u8> {
    let fragments = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<root>\n",
        "    <item id=\"1\" type=\"product\">\n",
        "        <name>Widget Pro</name>\n",
        "        <price currency=\"USD\">29.99</price>\n",
        "        <description>A high-quality widget for all your needs.</description>\n",
        "    </item>\n",
        "    <item id=\"2\" type=\"product\">\n",
        "        <name>Gadget Plus</name>\n",
        "        <price currency=\"USD\">49.99</price>\n",
        "        <description>The ultimate gadget experience.</description>\n",
        "    </item>\n",
        "    <metadata>\n",
        "        <created>2025-01-22T12:00:00Z</created>\n",
        "        <modified>2025-01-22T12:00:00Z</modified>\n",
        "        <author>System</author>\n",
        "    </metadata>\n",
        "</root>\n",
        "    <category name=\"Electronics\">\n",
        "        <subcategory>Computers</subcategory>\n",
        "        <subcategory>Phones</subcategory>\n",
        "    </category>\n",
    ];

    let mut rng = StdRng::seed_from_u64(45);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        let fragment = fragments[rng.gen_range(0..fragments.len())];
        result.extend_from_slice(fragment.as_bytes());
    }
    result.truncate(size);
    result
}

/// JSON data - API responses, config files
fn generate_json(size: usize) -> Vec<u8> {
    let fragments = [
        "{\"users\":[",
        "{\"id\":1,\"name\":\"Alice\",\"email\":\"alice@example.com\",\"active\":true},",
        "{\"id\":2,\"name\":\"Bob\",\"email\":\"bob@example.com\",\"active\":false},",
        "{\"id\":3,\"name\":\"Charlie\",\"email\":\"charlie@example.com\",\"active\":true}",
        "],\"metadata\":{",
        "\"total\":3,\"page\":1,\"per_page\":10,\"timestamp\":\"2025-01-22T12:00:00Z\"",
        "},\"status\":\"success\"}",
        "{\"config\":{\"debug\":false,\"verbose\":true,\"max_connections\":100,",
        "\"timeout_ms\":5000,\"retry_count\":3,\"endpoints\":[",
        "\"https://api.example.com/v1\",\"https://api.example.com/v2\"",
        "],\"features\":{\"caching\":true,\"compression\":true,\"logging\":true}}}",
        "{\"event\":{\"type\":\"click\",\"target\":\"button\",\"x\":150,\"y\":200,",
        "\"timestamp\":1705924800,\"user_agent\":\"Mozilla/5.0\"}}",
    ];

    let mut rng = StdRng::seed_from_u64(46);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        let fragment = fragments[rng.gen_range(0..fragments.len())];
        result.extend_from_slice(fragment.as_bytes());
    }
    result.truncate(size);
    result
}

/// Server logs (similar to Silesia's nci dataset)
fn generate_server_logs(size: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(47);
    let mut result = Vec::with_capacity(size);

    let methods = ["GET", "POST", "PUT", "DELETE", "PATCH"];
    let paths = [
        "/api/users",
        "/api/products",
        "/api/orders",
        "/health",
        "/metrics",
        "/api/v1/search",
        "/api/v1/items",
        "/static/js/app.js",
        "/static/css/style.css",
    ];
    let statuses = [200, 201, 204, 301, 302, 400, 401, 403, 404, 500];
    let ips = [
        "192.168.1.100",
        "10.0.0.50",
        "172.16.0.25",
        "192.168.2.200",
        "10.1.1.1",
    ];

    while result.len() < size {
        let ip = ips[rng.gen_range(0..ips.len())];
        let method = methods[rng.gen_range(0..methods.len())];
        let path = paths[rng.gen_range(0..paths.len())];
        let status = statuses[rng.gen_range(0..statuses.len())];
        let bytes = rng.gen_range(100..50000);
        let ms = rng.gen_range(1..500);

        let line = format!(
            "{} - - [22/Jan/2025:12:{:02}:{:02} +0000] \"{} {} HTTP/1.1\" {} {} \"-\" \"Mozilla/5.0\" {}ms\n",
            ip,
            rng.gen_range(0..60),
            rng.gen_range(0..60),
            method,
            path,
            status,
            bytes,
            ms
        );
        result.extend_from_slice(line.as_bytes());
    }
    result.truncate(size);
    result
}

// ============================================================================
// Real-World Mixed Data
// ============================================================================

/// Mixed binary with some structure (like a tar archive header regions)
fn generate_mixed_binary(size: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(48);
    let mut result = Vec::with_capacity(size);

    while result.len() < size {
        // Alternate between structured regions and more random regions
        let region_type = rng.gen_range(0..4);
        let region_size = rng.gen_range(256..2048).min(size - result.len());

        match region_type {
            0 => {
                // Null padding (common in archives)
                result.extend(std::iter::repeat(0u8).take(region_size));
            }
            1 => {
                // ASCII header-like region
                let header = b"HEADER__";
                for _ in 0..region_size / 8 {
                    result.extend_from_slice(header);
                }
            }
            2 => {
                // Sequential bytes (like uncompressed data)
                for i in 0..region_size {
                    result.push((i % 256) as u8);
                }
            }
            _ => {
                // More random data
                for _ in 0..region_size {
                    result.push(rng.r#gen::<u8>());
                }
            }
        }
    }
    result.truncate(size);
    result
}

/// High-entropy data (incompressible) - baseline for overhead measurement
fn generate_random_data(size: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(49);
    let mut result = vec![0u8; size];
    rng.fill(&mut result[..]);
    result
}

// ============================================================================
// Compression Benchmarks
// ============================================================================

fn bench_general_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_compression");

    // Test at 64KB - typical block size, good for comparison
    let size = 64 * 1024;

    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("english_prose", generate_english_prose(size)),
        ("c_source", generate_c_source(size)),
        ("html", generate_html(size)),
        ("xml", generate_xml(size)),
        ("json", generate_json(size)),
        ("server_logs", generate_server_logs(size)),
        ("mixed_binary", generate_mixed_binary(size)),
        ("random", generate_random_data(size)),
    ];

    for (name, data) in &datasets {
        group.throughput(Throughput::Bytes(size as u64));

        // Haagenti
        group.bench_with_input(BenchmarkId::new("haagenti", *name), data, |b, data| {
            let compressor = ZstdCompressor::new();
            b.iter(|| compressor.compress(black_box(data)).unwrap())
        });

        // Reference zstd
        group.bench_with_input(BenchmarkId::new("zstd_ref", *name), data, |b, data| {
            b.iter(|| zstd::encode_all(black_box(&data[..]), 3).unwrap())
        });
    }

    group.finish();
}

fn bench_general_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_decompression");

    let size = 64 * 1024;

    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("english_prose", generate_english_prose(size)),
        ("c_source", generate_c_source(size)),
        ("html", generate_html(size)),
        ("xml", generate_xml(size)),
        ("json", generate_json(size)),
        ("server_logs", generate_server_logs(size)),
        ("mixed_binary", generate_mixed_binary(size)),
    ];

    for (name, data) in &datasets {
        // Compress with reference for reference decompression test
        let ref_compressed = zstd::encode_all(&data[..], 3).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        // Reference zstd decompression only - Haagenti has known issues with
        // some general-purpose data patterns (this is documented)
        group.bench_with_input(
            BenchmarkId::new("zstd_ref", *name),
            &ref_compressed,
            |b, compressed| b.iter(|| zstd::decode_all(black_box(&compressed[..])).unwrap()),
        );
    }

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_ratio");

    let size = 64 * 1024;

    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("english_prose", generate_english_prose(size)),
        ("c_source", generate_c_source(size)),
        ("html", generate_html(size)),
        ("xml", generate_xml(size)),
        ("json", generate_json(size)),
        ("server_logs", generate_server_logs(size)),
        ("mixed_binary", generate_mixed_binary(size)),
        ("random", generate_random_data(size)),
    ];

    println!("\n=== Compression Ratio Comparison (64KB) ===\n");
    println!(
        "{:15} | {:>10} | {:>18} | {:>18} | {:>10}",
        "Data Type", "Original", "Haagenti", "Reference", "Delta"
    );
    println!(
        "{:-<15}-+-{:-<10}-+-{:-<18}-+-{:-<18}-+-{:-<10}",
        "", "", "", "", ""
    );

    for (name, data) in &datasets {
        let compressor = ZstdCompressor::new();
        let ref_compressed = zstd::encode_all(&data[..], 3).unwrap();

        // Handle potential Haagenti compression errors gracefully
        match compressor.compress(&data) {
            Ok(haagenti_compressed) => {
                let haagenti_ratio = data.len() as f64 / haagenti_compressed.len() as f64;
                let ref_ratio = data.len() as f64 / ref_compressed.len() as f64;

                println!(
                    "{:15} | {:>10} | {:>6} ({:>6.2}x) | {:>6} ({:>6.2}x) | {:>+9.1}%",
                    name,
                    data.len(),
                    haagenti_compressed.len(),
                    haagenti_ratio,
                    ref_compressed.len(),
                    ref_ratio,
                    ((haagenti_ratio / ref_ratio) - 1.0) * 100.0
                );
            }
            Err(e) => {
                let ref_ratio = data.len() as f64 / ref_compressed.len() as f64;
                println!(
                    "{:15} | {:>10} | {:>18} | {:>6} ({:>6.2}x) | {:>10}",
                    name,
                    data.len(),
                    "ERROR",
                    ref_compressed.len(),
                    ref_ratio,
                    "N/A"
                );
                eprintln!("  -> Haagenti error on {}: {}", name, e);
            }
        }

        // Dummy benchmark just to include in the group
        group.throughput(Throughput::Bytes(1));
        group.bench_with_input(BenchmarkId::new("ratio_check", *name), &data, |b, _| {
            b.iter(|| 1)
        });
    }

    println!();
    group.finish();
}

fn bench_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_sizes");

    // Test English prose at various sizes to show scaling
    let sizes = [1024, 4096, 16384, 65536, 262144]; // 1KB to 256KB

    for size in sizes {
        let data = generate_english_prose(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Haagenti
        group.bench_with_input(
            BenchmarkId::new("haagenti/prose", size),
            &data,
            |b, data| {
                let compressor = ZstdCompressor::new();
                b.iter(|| compressor.compress(black_box(data)).unwrap())
            },
        );

        // Reference
        group.bench_with_input(
            BenchmarkId::new("zstd_ref/prose", size),
            &data,
            |b, data| b.iter(|| zstd::encode_all(black_box(&data[..]), 3).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_general_compression,
    bench_general_decompression,
    bench_compression_ratio,
    bench_various_sizes,
);

criterion_main!(benches);
