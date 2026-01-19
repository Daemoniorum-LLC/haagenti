//! Minimal test: encode fixed sequences and compare with reference
//!
//! This tests FSE encoding in isolation by creating the exact same frame
//! structure as reference and comparing byte-by-byte.

use std::io::Cursor;

fn main() {
    println!("=== Minimal FSE Test ===\n");
    
    // Create input that produces a 2-sequence pattern
    // Pattern: "ABCD" * 10 + "XXXX" + "EFGH" * 5 = 64 bytes
    // This should produce 2 sequences: one for ABCD matches, one for EFGH matches

    let mut input = Vec::new();
    for _ in 0..10 { input.extend_from_slice(b"ABCD"); }
    input.extend_from_slice(b"XXXX");
    for _ in 0..5 { input.extend_from_slice(b"EFGH"); }
    println!("Input: {} bytes of 'ABCD' repeated", input.len());
    
    // Get reference compression
    let ref_compressed = zstd::encode_all(Cursor::new(&input), 1).unwrap();
    println!("\nReference compressed: {} bytes", ref_compressed.len());
    println!("Reference hex: {:02x?}", ref_compressed);
    
    // Parse reference frame to find sequences section
    // Frame header: magic (4) + FHD (1) + optional window/content
    let fhd = ref_compressed[4];
    let single_segment = (fhd & 0x20) != 0;
    let content_size_flag = (fhd >> 6) & 0x03;
    
    let mut pos = 5;
    if !single_segment {
        pos += 1; // window descriptor
    }
    pos += match content_size_flag {
        0 => if single_segment { 1 } else { 0 },
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    
    println!("Block starts at: {}", pos);
    
    // Block header (3 bytes)
    let block_header = ref_compressed[pos] as u32 
        | ((ref_compressed[pos+1] as u32) << 8) 
        | ((ref_compressed[pos+2] as u32) << 16);
    let last_block = (block_header & 1) != 0;
    let block_type = (block_header >> 1) & 0x03;
    let block_size = block_header >> 3;
    
    println!("Block: last={}, type={}, size={}", last_block, block_type, block_size);
    
    if block_type == 2 {
        // Compressed block
        let block_start = pos + 3;
        let block_data = &ref_compressed[block_start..block_start + block_size as usize];
        
        // Parse literals header
        let lit_header = block_data[0];
        let lit_type = lit_header & 0x03;
        let size_format = (lit_header >> 2) & 0x03;
        
        let (lit_header_size, regen_size, comp_size) = match (lit_type, size_format) {
            (0, 0) | (1, 0) => (1, (lit_header >> 3) as usize, 0),
            (0, 1) | (1, 1) => {
                let size = ((lit_header >> 4) as usize) | ((block_data[1] as usize) << 4);
                (2, size, 0)
            },
            (2, 0) | (3, 0) => {
                let sizes = ((lit_header >> 4) as usize) 
                    | ((block_data[1] as usize) << 4)
                    | ((block_data[2] as usize) << 12);
                (3, sizes & 0x3FF, (sizes >> 10) & 0x3FF)
            },
            _ => panic!("Complex literals header"),
        };
        
        let literals_size = if comp_size > 0 { comp_size } else { regen_size };
        println!("Literals: type={}, regen={}, comp={}, header_size={}", 
                 lit_type, regen_size, comp_size, lit_header_size);
        
        // Sequences section
        let seq_start = lit_header_size + literals_size;
        let seq_data = &block_data[seq_start..];
        
        let seq_count = seq_data[0] as usize;
        let (count, header_len) = if seq_count < 128 {
            (seq_count, 1)
        } else if seq_count < 255 {
            (((seq_count - 128) << 8) | seq_data[1] as usize, 2)
        } else {
            ((seq_data[1] as usize) | ((seq_data[2] as usize) << 8) + 0x7F00, 3)
        };
        
        println!("\nSequences: {} count", count);
        
        let mode_byte = seq_data[header_len];
        let ll_mode = (mode_byte >> 6) & 0x03;
        let of_mode = (mode_byte >> 4) & 0x03;
        let ml_mode = (mode_byte >> 2) & 0x03;
        println!("Mode byte: {:02x} (LL={}, OF={}, ML={})", mode_byte, ll_mode, of_mode, ml_mode);
        
        let fse_start = header_len + 1;
        let fse_data = &seq_data[fse_start..];
        println!("FSE bitstream: {} bytes = {:02x?}", fse_data.len(), fse_data);
        
        // Print binary representation
        print!("Binary (LSB first per byte): ");
        for b in fse_data {
            print!("{:08b} ", b.reverse_bits());
        }
        println!();
    }
    
    // Show what sequences we generate
    use haagenti_zstd::compress::{LazyMatchFinder, block::matches_to_sequences, EncodedSequence};

    let mut mf = LazyMatchFinder::new(16);
    let matches = mf.find_matches(&input);
    println!("\n=== Our Match Finding ===");
    println!("Matches: {}", matches.len());
    for (i, m) in matches.iter().enumerate() {
        println!("  Match[{}]: pos={}, len={}, offset={}", i, m.position, m.length, m.offset);
    }

    let (literals, sequences) = matches_to_sequences(&input, &matches);
    println!("Sequences: {}", sequences.len());
    for (i, s) in sequences.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(s);
        println!("  Seq[{}]: ll={}, offset={}, ml={}", i, s.literal_length, s.offset, s.match_length);
        println!("    Encoded: ll_code={}, of_code={}, ml_code={}", enc.ll_code, enc.of_code, enc.ml_code);
        println!("    Extras: ll={} ({}b), ml={} ({}b), of={} ({}b)",
                 enc.ll_extra, enc.ll_bits, enc.ml_extra, enc.ml_bits, enc.of_extra, enc.of_bits);
    }

    // Now compress with our encoder
    use haagenti_core::Compressor;
    use haagenti_zstd::ZstdCompressor;

    let compressor = ZstdCompressor::new();
    let our_compressed = compressor.compress(&input).unwrap();
    
    println!("\n=== Our Compression ===");
    println!("Our compressed: {} bytes", our_compressed.len());
    println!("Our hex: {:02x?}", our_compressed);
    
    // Try to decode with reference
    match zstd::decode_all(Cursor::new(&our_compressed)) {
        Ok(decoded) if decoded == input => println!("\nReference decode: SUCCESS!"),
        Ok(_) => println!("\nReference decode: MISMATCH"),
        Err(e) => println!("\nReference decode: FAILED - {:?}", e),
    }

    // Decode reference's FSE bitstream to see what sequences it contains
    println!("\n=== Decoding Reference FSE Bitstream ===");
    use haagenti_zstd::fse::{FseTable, FseDecoder, BitReader,
        LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
        OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
        MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG};
    use haagenti_zstd::block::{LITERAL_LENGTH_BASELINE, MATCH_LENGTH_BASELINE};

    let ref_fse = vec![0x00u8, 0xe1, 0x27, 0x1d, 0x11];
    let mut bits = BitReader::new(&ref_fse);
    bits.init_from_end().expect("init");

    let ll_table = FseTable::from_predefined(&LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG).unwrap();
    let of_table = FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(&MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG).unwrap();

    let mut ll_dec = FseDecoder::new(&ll_table);
    let mut of_dec = FseDecoder::new(&of_table);
    let mut ml_dec = FseDecoder::new(&ml_table);

    ll_dec.init_state(&mut bits).unwrap();
    of_dec.init_state(&mut bits).unwrap();
    ml_dec.init_state(&mut bits).unwrap();

    println!("Initial states: LL={}, OF={}, ML={}",
             ll_dec.peek_symbol(), of_dec.peek_symbol(), ml_dec.peek_symbol());
    println!("Bits remaining: {}", bits.bits_remaining());

    bits.switch_to_lsb_mode().unwrap();
    println!("After LSB switch, bits remaining: {}", bits.bits_remaining());

    // Decode seq 0 (not last)
    let ll_code = ll_dec.peek_symbol();
    let of_code = of_dec.peek_symbol();
    let ml_code = ml_dec.peek_symbol();

    fn offset_code_extra_bits(code: u8) -> u8 {
        if code == 0 { 0 } else { code }
    }

    println!("\n--- Decoding Seq 0 ---");
    println!("Initial states give codes: ll={}, of={}, ml={}", ll_code, of_code, ml_code);

    // Calculate expected extra bits
    let ll_bits_needed = if ll_code < LITERAL_LENGTH_BASELINE.len() as u8 {
        LITERAL_LENGTH_BASELINE[ll_code as usize].0
    } else { 0 };
    let ml_bits_needed = ml_dec.peek_seq_extra_bits();
    let of_bits_needed = offset_code_extra_bits(of_code);

    println!("Extra bits needed: ll={}, ml={}, of={}", ll_bits_needed, ml_bits_needed, of_bits_needed);
    println!("Bits remaining before extras: {}", bits.bits_remaining());

    // Read extras in LL, ML, OF order
    let ll_extra = if ll_bits_needed > 0 {
        bits.read_bits(ll_bits_needed as usize).unwrap_or(999)
    } else { 0 };
    println!("After LL extra (read {} bits): ll_extra={}, bits_remaining={}",
             ll_bits_needed, ll_extra, bits.bits_remaining());

    let ml_seq_base = ml_dec.peek_seq_base();
    let ml_extra = if ml_bits_needed > 0 {
        bits.read_bits(ml_bits_needed as usize).unwrap_or(999)
    } else { 0 };
    println!("After ML extra (read {} bits): ml_extra={}, bits_remaining={}",
             ml_bits_needed, ml_extra, bits.bits_remaining());

    let of_extra = if of_bits_needed > 0 {
        bits.read_bits(of_bits_needed as usize).unwrap_or(999)
    } else { 0 };
    println!("After OF extra (read {} bits): of_extra={}, bits_remaining={}",
             of_bits_needed, of_extra, bits.bits_remaining());

    println!("\nSeq 0 decoded values:");
    println!("  ll_code={} + ll_extra={} → ll=?", ll_code, ll_extra);
    println!("  ml_code={} + ml_extra={} (base={}) → ml={}", ml_code, ml_extra, ml_seq_base, ml_seq_base + ml_extra);
    println!("  of_code={} + of_extra={} → offset_value={}", of_code, of_extra, (1u64 << of_code) + of_extra as u64);

    // FSE update
    println!("\n--- FSE State Update ---");
    println!("Before FSE update, bits remaining: {}", bits.bits_remaining());

    // Get num_bits from decoder for each state update
    println!("Updating LL state...");
    ll_dec.update_state(&mut bits).unwrap();
    println!("After LL update: new_symbol={}, bits_remaining={}", ll_dec.peek_symbol(), bits.bits_remaining());

    println!("Updating ML state...");
    ml_dec.update_state(&mut bits).unwrap();
    println!("After ML update: new_symbol={}, bits_remaining={}", ml_dec.peek_symbol(), bits.bits_remaining());

    println!("Updating OF state...");
    of_dec.update_state(&mut bits).unwrap();
    println!("After OF update: new_symbol={}, bits_remaining={}", of_dec.peek_symbol(), bits.bits_remaining());

    // Decode seq 1 (last)
    let ll_code2 = ll_dec.peek_symbol();
    let of_code2 = of_dec.peek_symbol();
    let ml_code2 = ml_dec.peek_symbol();

    println!("\n--- Decoding Seq 1 (last) ---");
    println!("States give codes: ll={}, of={}, ml={}", ll_code2, of_code2, ml_code2);
    println!("Bits remaining: {}", bits.bits_remaining());

    // Read seq 1 extras
    let ll_bits2 = if ll_code2 < LITERAL_LENGTH_BASELINE.len() as u8 {
        LITERAL_LENGTH_BASELINE[ll_code2 as usize].0
    } else { 0 };
    let ll_extra2 = if ll_bits2 > 0 { bits.read_bits(ll_bits2 as usize).unwrap_or(999) } else { 0 };

    let of_bits2 = offset_code_extra_bits(of_code2);
    let of_extra2 = if of_bits2 > 0 { bits.read_bits(of_bits2 as usize).unwrap_or(999) } else { 0 };
    println!("Seq 1 extras: ll={}, of={}", ll_extra2, of_extra2);
    println!("Final bits remaining: {}", bits.bits_remaining());

    // Now let's also decode OUR bitstream to compare
    println!("\n\n=== Decoding OUR FSE Bitstream ===");
    let our_fse = vec![0xe7u8, 0x88, 0x9b, 0x74, 0x44];
    let mut our_bits = BitReader::new(&our_fse);
    our_bits.init_from_end().expect("init");

    let mut our_ll_dec = FseDecoder::new(&ll_table);
    let mut our_of_dec = FseDecoder::new(&of_table);
    let mut our_ml_dec = FseDecoder::new(&ml_table);

    our_ll_dec.init_state(&mut our_bits).unwrap();
    our_of_dec.init_state(&mut our_bits).unwrap();
    our_ml_dec.init_state(&mut our_bits).unwrap();

    println!("Initial states: LL={}, OF={}, ML={}",
             our_ll_dec.peek_symbol(), our_of_dec.peek_symbol(), our_ml_dec.peek_symbol());
    println!("Bits remaining after init: {}", our_bits.bits_remaining());

    our_bits.switch_to_lsb_mode().unwrap();
    println!("After LSB switch, bits remaining: {}", our_bits.bits_remaining());

    // Decode our seq 0
    let our_ll0 = our_ll_dec.peek_symbol();
    let our_of0 = our_of_dec.peek_symbol();
    let our_ml0 = our_ml_dec.peek_symbol();
    println!("\nOur Seq 0 codes: ll={}, of={}, ml={}", our_ll0, our_of0, our_ml0);

    let our_ll0_bits = if our_ll0 < LITERAL_LENGTH_BASELINE.len() as u8 {
        LITERAL_LENGTH_BASELINE[our_ll0 as usize].0
    } else { 0 };
    let our_ml0_bits = our_ml_dec.peek_seq_extra_bits();
    let our_of0_bits = offset_code_extra_bits(our_of0);

    let our_ll0_extra = if our_ll0_bits > 0 { our_bits.read_bits(our_ll0_bits as usize).unwrap_or(999) } else { 0 };
    let our_ml0_extra = if our_ml0_bits > 0 { our_bits.read_bits(our_ml0_bits as usize).unwrap_or(999) } else { 0 };
    let our_of0_extra = if our_of0_bits > 0 { our_bits.read_bits(our_of0_bits as usize).unwrap_or(999) } else { 0 };

    println!("Our Seq 0 extras: ll={}, ml={}, of={}", our_ll0_extra, our_ml0_extra, our_of0_extra);
    println!("Bits remaining: {}", our_bits.bits_remaining());

    // FSE updates for our bitstream
    our_ll_dec.update_state(&mut our_bits).unwrap();
    our_ml_dec.update_state(&mut our_bits).unwrap();
    our_of_dec.update_state(&mut our_bits).unwrap();

    println!("After FSE updates: bits_remaining={}", our_bits.bits_remaining());

    // Our seq 1
    let our_ll1 = our_ll_dec.peek_symbol();
    let our_of1 = our_of_dec.peek_symbol();
    let our_ml1 = our_ml_dec.peek_symbol();
    println!("Our Seq 1 codes: ll={}, of={}, ml={}", our_ll1, our_of1, our_ml1);
}
