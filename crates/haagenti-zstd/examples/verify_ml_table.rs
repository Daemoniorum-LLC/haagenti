//! Verify ML table against reference zstd

fn main() {
    println!("=== Reference ML_defaultDTable Analysis ===\n");
    
    // Reference ML_defaultDTable from zstd source
    // Format: (nextState, nbAddBits, nbBits, baseVal)
    let ref_table: [(u16, u8, u8, u32); 64] = [
        (  0,  0,  6,    3), (  0,  0,  4,    4),  // states 0,1
        ( 32,  0,  5,    5), (  0,  0,  5,    6),  // states 2,3
        (  0,  0,  5,    8), (  0,  0,  5,    9),  // states 4,5
        (  0,  0,  5,   11), (  0,  0,  6,   13),  // states 6,7
        (  0,  0,  6,   16), (  0,  0,  6,   19),  // states 8,9
        (  0,  0,  6,   22), (  0,  0,  6,   25),  // states 10,11
        (  0,  0,  6,   28), (  0,  0,  6,   31),  // states 12,13
        (  0,  0,  6,   34), (  0,  1,  6,   37),  // states 14,15
        (  0,  1,  6,   41), (  0,  2,  6,   47),  // states 16,17
        (  0,  3,  6,   59), (  0,  4,  6,   83),  // states 18,19
        (  0,  7,  6,  131), (  0,  9,  6,  515),  // states 20,21
        ( 16,  0,  4,    4), (  0,  0,  4,    5),  // states 22,23
        ( 32,  0,  5,    6), (  0,  0,  5,    7),  // states 24,25
        ( 32,  0,  5,    9), (  0,  0,  5,   10),  // states 26,27
        (  0,  0,  6,   12), (  0,  0,  6,   15),  // states 28,29
        (  0,  0,  6,   18), (  0,  0,  6,   21),  // states 30,31
        (  0,  0,  6,   24), (  0,  0,  6,   27),  // states 32,33
        (  0,  0,  6,   30), (  0,  0,  6,   33),  // states 34,35
        (  0,  1,  6,   35), (  0,  1,  6,   39),  // states 36,37
        (  0,  2,  6,   43), (  0,  3,  6,   51),  // states 38,39
        (  0,  4,  6,   67), (  0,  5,  6,   99),  // states 40,41
        (  0,  8,  6,  259), ( 32,  0,  4,    4),  // states 42,43
        ( 48,  0,  4,    4), ( 16,  0,  4,    5),  // states 44,45
        ( 32,  0,  5,    7), ( 32,  0,  5,    8),  // states 46,47
        ( 32,  0,  5,   10), ( 32,  0,  5,   11),  // states 48,49
        (  0,  0,  6,   14), (  0,  0,  6,   17),  // states 50,51
        (  0,  0,  6,   20), (  0,  0,  6,   23),  // states 52,53
        (  0,  0,  6,   26), (  0,  0,  6,   29),  // states 54,55
        (  0,  0,  6,   32), (  0, 16,  6,65539),  // states 56,57
        (  0, 15,  6,32771), (  0, 14,  6,16387),  // states 58,59
        (  0, 13,  6, 8195), (  0, 12,  6, 4099),  // states 60,61
        (  0, 11,  6, 2051), (  0, 10,  6, 1027),  // states 62,63
    ];
    
    // ML baseline table from RFC 8878
    let ml_baselines: [(u32, u8); 53] = [
        (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),  // 0-7
        (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0),  // 8-15
        (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0),  // 16-23
        (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0),  // 24-31
        (35, 1), (37, 1), (39, 2), (43, 2), (47, 3), (51, 3), (59, 4), (67, 4),  // 32-39
        (83, 5), (99, 5), (131, 7), (259, 8), (515, 9), (1027, 10), (2051, 11),  // 40-46
        (4099, 12), (8195, 13), (16387, 14), (32771, 15), (65539, 16), (131075, 17), // 47-52
    ];
    
    println!("State -> ML Code mapping from reference (baseVal -> code lookup):");
    for state in 0..64 {
        let (_, nb_add_bits, nb_bits, base_val) = ref_table[state];
        
        // Find which ML code has this baseline and extra bits
        let ml_code = ml_baselines.iter()
            .position(|&(base, bits)| base == base_val && bits == nb_add_bits)
            .map(|c| c as u8);
        
        if let Some(code) = ml_code {
            if code >= 40 {
                println!("  State {:2} → ML code {:2} (baseVal={:6}, {} extra bits)",
                         state, code, base_val, nb_add_bits);
            }
        } else {
            // For codes that don't directly match, find by baseVal
            let code_by_base = ml_baselines.iter()
                .position(|&(base, _)| base == base_val);
            if let Some(c) = code_by_base {
                if c >= 40 {
                    println!("  State {:2} → ML code {:2} (baseVal={:6}, {} extra bits) [by base]",
                             state, c, base_val, nb_add_bits);
                }
            }
        }
    }
    
    // Specifically check state 42
    println!("\n=== State 42 Analysis ===");
    let (next, nb_add, nb_bits, base) = ref_table[42];
    println!("Reference state 42: nextState={}, nbAddBits={}, nbBits={}, baseVal={}",
             next, nb_add, nb_bits, base);
    
    // What ML code has baseline 259 and 8 extra bits?
    let ml_43 = ml_baselines[43];
    println!("ML code 43: baseline={}, extra_bits={}", ml_43.0, ml_43.1);
    println!("Match! State 42 → ML code 43");
    
    // For 500 bytes: 4 literals + 496 match
    println!("\n=== For match_length 496 ===");
    println!("ML code 43: 259 + extra(8 bits)");
    println!("Need: 496 = 259 + extra → extra = 237 = 0x{:02x}", 496 - 259);
    
    // Now print what our table has
    println!("\n=== Our Current Table ===");
    let our_table = haagenti_zstd::fse::FseTable::from_predefined(
        &haagenti_zstd::fse::MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        haagenti_zstd::fse::MATCH_LENGTH_ACCURACY_LOG,
    ).unwrap();
    
    for state in [41, 42, 43, 62, 63] {
        let entry = our_table.decode(state);
        println!("  Our state {:2} → symbol {:2}", state, entry.symbol);
    }
}
