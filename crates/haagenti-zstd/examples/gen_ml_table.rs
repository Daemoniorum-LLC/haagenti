//! Generate correct ML_PREDEFINED_TABLE from zstd reference

fn main() {
    // Reference ML_defaultDTable from zstd source
    // Format: (nextState, nbAddBits, nbBits, baseVal)
    let ref_table: [(u16, u8, u8, u32); 64] = [
        (0, 0, 6, 3),
        (0, 0, 4, 4), // states 0,1
        (32, 0, 5, 5),
        (0, 0, 5, 6), // states 2,3
        (0, 0, 5, 8),
        (0, 0, 5, 9), // states 4,5
        (0, 0, 5, 11),
        (0, 0, 6, 13), // states 6,7
        (0, 0, 6, 16),
        (0, 0, 6, 19), // states 8,9
        (0, 0, 6, 22),
        (0, 0, 6, 25), // states 10,11
        (0, 0, 6, 28),
        (0, 0, 6, 31), // states 12,13
        (0, 0, 6, 34),
        (0, 1, 6, 37), // states 14,15
        (0, 1, 6, 41),
        (0, 2, 6, 47), // states 16,17
        (0, 3, 6, 59),
        (0, 4, 6, 83), // states 18,19
        (0, 7, 6, 131),
        (0, 9, 6, 515), // states 20,21
        (16, 0, 4, 4),
        (0, 0, 4, 5), // states 22,23
        (32, 0, 5, 6),
        (0, 0, 5, 7), // states 24,25
        (32, 0, 5, 9),
        (0, 0, 5, 10), // states 26,27
        (0, 0, 6, 12),
        (0, 0, 6, 15), // states 28,29
        (0, 0, 6, 18),
        (0, 0, 6, 21), // states 30,31
        (0, 0, 6, 24),
        (0, 0, 6, 27), // states 32,33
        (0, 0, 6, 30),
        (0, 0, 6, 33), // states 34,35
        (0, 1, 6, 35),
        (0, 1, 6, 39), // states 36,37
        (0, 2, 6, 43),
        (0, 3, 6, 51), // states 38,39
        (0, 4, 6, 67),
        (0, 5, 6, 99), // states 40,41
        (0, 8, 6, 259),
        (32, 0, 4, 4), // states 42,43
        (48, 0, 4, 4),
        (16, 0, 4, 5), // states 44,45
        (32, 0, 5, 7),
        (32, 0, 5, 8), // states 46,47
        (32, 0, 5, 10),
        (32, 0, 5, 11), // states 48,49
        (0, 0, 6, 14),
        (0, 0, 6, 17), // states 50,51
        (0, 0, 6, 20),
        (0, 0, 6, 23), // states 52,53
        (0, 0, 6, 26),
        (0, 0, 6, 29), // states 54,55
        (0, 0, 6, 32),
        (0, 16, 6, 65539), // states 56,57
        (0, 15, 6, 32771),
        (0, 14, 6, 16387), // states 58,59
        (0, 13, 6, 8195),
        (0, 12, 6, 4099), // states 60,61
        (0, 11, 6, 2051),
        (0, 10, 6, 1027), // states 62,63
    ];

    // ML_BASELINE_TABLE format: (extra_bits, baseline)
    let ml_baseline: [(u8, u32); 53] = [
        // Codes 0-31: No extra bits, match_length = baseline (3-34)
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 14),
        (0, 15),
        (0, 16),
        (0, 17),
        (0, 18),
        (0, 19),
        (0, 20),
        (0, 21),
        (0, 22),
        (0, 23),
        (0, 24),
        (0, 25),
        (0, 26),
        (0, 27),
        (0, 28),
        (0, 29),
        (0, 30),
        (0, 31),
        (0, 32),
        (0, 33),
        (0, 34),
        // Codes 32-35: 1 extra bit each
        (1, 35),
        (1, 37),
        (1, 39),
        (1, 41),
        // Codes 36-37: 2 extra bits each
        (2, 43),
        (2, 47),
        // Codes 38-39: 3 extra bits each
        (3, 51),
        (3, 59),
        // Codes 40-41: 4 extra bits each
        (4, 67),
        (4, 83),
        // Code 42: 5 extra bits
        (5, 99),
        // Code 43: 7 extra bits
        (7, 131),
        // Code 44: 8 extra bits
        (8, 259),
        // Code 45: 9 extra bits
        (9, 515),
        // Code 46-52
        (10, 1027),
        (11, 2051),
        (12, 4099),
        (13, 8195),
        (14, 16387),
        (15, 32771),
        (16, 65539),
    ];

    println!("/// Hardcoded Match Length decode table from zstd's seqSymbolTable_ML_defaultDistribution.");
    println!("/// Each entry is (symbol, nbBits, baseline) for FSE state transitions.");
    println!("/// These values are taken directly from zstd's reference implementation.");
    println!("const ML_PREDEFINED_TABLE: [(u8, u8, u16); 64] = [");

    for chunk in (0..64).collect::<Vec<_>>().chunks(4) {
        let entries: Vec<String> = chunk
            .iter()
            .map(|&state| {
                let (next_state, nb_add_bits, nb_bits, base_val) = ref_table[state];

                // Find ML code that matches (nb_add_bits, base_val)
                let ml_code = ml_baseline
                    .iter()
                    .position(|&(extra_bits, baseline)| {
                        extra_bits == nb_add_bits && baseline == base_val
                    })
                    .unwrap_or_else(|| panic!("No ML code found for state {} (nbAddBits={}, baseVal={})",
                        state, nb_add_bits, base_val)) as u8;

                // Calculate baseline (nextState from zstd)
                let baseline = next_state;

                format!("({:>2}, {}, {:>2})", ml_code, nb_bits, baseline)
            })
            .collect();

        let comment = format!("// states {}-{}", chunk[0], chunk[chunk.len() - 1]);
        println!("    {}, {}", entries.join(", "), comment);
    }

    println!("];");

    // Also verify state 42 specifically
    println!("\n=== Verification ===");
    let (next_state, nb_add_bits, nb_bits, base_val) = ref_table[42];
    println!(
        "State 42: nextState={}, nbAddBits={}, nbBits={}, baseVal={}",
        next_state, nb_add_bits, nb_bits, base_val
    );

    let ml_code = ml_baseline
        .iter()
        .position(|&(extra_bits, baseline)| extra_bits == nb_add_bits && baseline == base_val)
        .map(|c| c as u8);
    println!(
        "ML code for (nbAddBits={}, baseVal={}): {:?}",
        nb_add_bits, base_val, ml_code
    );
}
