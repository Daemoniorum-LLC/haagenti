use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader};
use std::fs::File;
use std::io::BufReader;

fn main() {
    // Test inline format model (total_fragments=0 in header but data present)
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/crook/models/smollm2-135m-hct3/model.embed_tokens.weight.hct".to_string()
    });
    println!("Testing: {}", path);

    let file = File::open(&path).expect("Failed to open file");
    let reader = BufReader::new(file);

    match HoloTensorReader::new(reader) {
        Ok(mut holo_reader) => {
            println!("Header parsed successfully");
            let header = holo_reader.header();
            println!("  Encoding: {:?}", header.encoding);
            println!("  Compression: {:?}", header.compression);
            println!("  Shape: {:?}", header.shape);
            println!("  Total fragments (header): {}", header.total_fragments);
            println!(
                "  Total fragments (effective): {}",
                holo_reader.total_fragments()
            );
            println!("  Is inline format: {}", holo_reader.is_inline_format());

            match holo_reader.read_all() {
                Ok((header, fragments)) => {
                    println!("read_all succeeded: {} fragments", fragments.len());

                    let mut decoder = HoloTensorDecoder::new(header.clone());
                    for frag in fragments {
                        println!(
                            "  Fragment {}: {} bytes, flags={}",
                            frag.index,
                            frag.data.len(),
                            frag.flags
                        );
                        if let Err(e) = decoder.add_fragment(frag) {
                            println!("  ERROR adding fragment: {:?}", e);
                        }
                    }

                    match decoder.reconstruct() {
                        Ok(data) => {
                            println!("Reconstructed {} f32 values", data.len());
                            println!("First 5 values: {:?}", &data[..5.min(data.len())]);
                        }
                        Err(e) => println!("Reconstruction failed: {:?}", e),
                    }
                }
                Err(e) => println!("read_all failed: {:?}", e),
            }
        }
        Err(e) => println!("Failed to create reader: {:?}", e),
    }
}
