//! Verify our predefined FSE tables match reference zstd.

use haagenti_zstd::fse::{
    FseTable, LITERAL_LENGTH_ACCURACY_LOG, LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
    MATCH_LENGTH_ACCURACY_LOG, MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
    OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== Verifying Predefined FSE Tables ===\n");

    // Reference zstd's predefined tables for verification
    // These are the decode table entries (state -> symbol, num_bits, baseline)

    // From zstd's seqSymbolTable_LL_defaultDistribution
    // Format: (symbol, num_bits, baseline) for each state
    let ref_ll_table: [(u8, u8, u16); 64] = [
        (0, 4, 0),
        (0, 4, 16),
        (1, 5, 32),
        (3, 5, 0), // states 0-3
        (4, 5, 0),
        (6, 5, 0),
        (7, 5, 0),
        (9, 5, 0), // states 4-7
        (10, 5, 0),
        (12, 5, 0),
        (14, 6, 0),
        (16, 5, 0), // states 8-11
        (18, 5, 0),
        (19, 5, 0),
        (21, 5, 0),
        (22, 5, 0), // states 12-15
        (24, 5, 0),
        (25, 6, 0),
        (26, 5, 0),
        (27, 6, 0), // states 16-19
        (29, 6, 0),
        (31, 6, 0),
        (0, 4, 32),
        (1, 4, 0), // states 20-23
        (2, 5, 0),
        (4, 5, 32),
        (5, 5, 0),
        (7, 5, 32), // states 24-27
        (8, 5, 0),
        (10, 5, 32),
        (11, 5, 0),
        (13, 6, 0), // states 28-31
        (16, 5, 32),
        (17, 5, 0),
        (19, 5, 32),
        (20, 5, 0), // states 32-35
        (22, 5, 32),
        (23, 5, 0),
        (25, 4, 0),
        (25, 4, 16), // states 36-39
        (26, 5, 32),
        (28, 6, 0),
        (30, 6, 0),
        (0, 4, 48), // states 40-43
        (1, 4, 16),
        (2, 5, 32),
        (3, 5, 32),
        (5, 5, 32), // states 44-47
        (6, 5, 32),
        (8, 5, 32),
        (9, 5, 32),
        (11, 5, 32), // states 48-51
        (12, 5, 32),
        (15, 6, 0),
        (17, 5, 32),
        (18, 5, 32), // states 52-55
        (20, 5, 32),
        (21, 5, 32),
        (23, 5, 32),
        (24, 5, 32), // states 56-59
        (35, 6, 0),
        (34, 6, 0),
        (33, 6, 0),
        (32, 6, 0), // states 60-63
    ];

    // From zstd's seqSymbolTable_ML_defaultDistribution
    let ref_ml_table: [(u8, u8, u16); 64] = [
        (0, 6, 0),
        (1, 4, 0),
        (2, 5, 32),
        (3, 5, 0),
        (5, 5, 0),
        (6, 5, 0),
        (8, 5, 0),
        (10, 6, 0),
        (13, 6, 0),
        (16, 6, 0),
        (19, 6, 0),
        (22, 6, 0),
        (25, 6, 0),
        (28, 6, 0),
        (31, 6, 0),
        (33, 6, 0),
        (35, 6, 0),
        (37, 6, 0),
        (39, 6, 0),
        (41, 6, 0),
        (43, 6, 0),
        (45, 6, 0),
        (1, 4, 16),
        (1, 4, 32),
        (1, 4, 48),
        (2, 4, 0),
        (2, 4, 16),
        (3, 5, 32),
        (4, 5, 0),
        (5, 5, 32),
        (6, 5, 32),
        (7, 5, 0),
        (7, 5, 32),
        (9, 6, 0),
        (12, 6, 0),
        (15, 6, 0),
        (18, 6, 0),
        (21, 6, 0),
        (24, 6, 0),
        (27, 6, 0),
        (30, 6, 0),
        (32, 6, 0),
        (34, 6, 0),
        (36, 6, 0),
        (38, 6, 0),
        (40, 6, 0),
        (42, 6, 0),
        (44, 6, 0),
        (47, 6, 0),
        (49, 6, 0),
        (52, 6, 0),
        (2, 4, 32),
        (4, 4, 0),
        (4, 4, 16),
        (4, 4, 32),
        (4, 4, 48),
        (8, 5, 32),
        (9, 4, 0),
        (11, 6, 0),
        (14, 6, 0),
        (17, 6, 0),
        (20, 6, 0),
        (23, 6, 0),
        (26, 6, 0),
    ];

    // From zstd's seqSymbolTable_OF_defaultDistribution
    let ref_of_table: [(u8, u8, u16); 32] = [
        (0, 5, 0),
        (6, 4, 0),
        (9, 5, 0),
        (15, 5, 0),
        (21, 5, 0),
        (3, 5, 0),
        (7, 4, 0),
        (12, 5, 0),
        (18, 5, 0),
        (23, 5, 0),
        (5, 5, 0),
        (8, 4, 0),
        (14, 5, 0),
        (20, 5, 0),
        (2, 5, 0),
        (7, 4, 16),
        (11, 5, 0),
        (17, 5, 0),
        (22, 5, 0),
        (4, 5, 0),
        (8, 4, 16),
        (13, 5, 0),
        (19, 5, 0),
        (1, 5, 0),
        (6, 4, 16),
        (10, 5, 0),
        (16, 5, 0),
        (28, 5, 0),
        (27, 5, 0),
        (26, 5, 0),
        (25, 5, 0),
        (24, 5, 0),
    ];

    // Build our tables
    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    )
    .unwrap();

    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    )
    .unwrap();

    let of_table =
        FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();

    // Compare LL table
    println!("=== LL Table (64 states) ===");
    let mut ll_match = true;
    #[allow(clippy::needless_range_loop)]
    for state in 0..64 {
        let ours = ll_table.decode(state);
        let (ref_sym, ref_bits, ref_base) = ref_ll_table[state];

        if ours.symbol != ref_sym || ours.num_bits != ref_bits || ours.baseline != ref_base {
            println!(
                "MISMATCH at state {}: ours=({}, {}, {}), ref=({}, {}, {})",
                state, ours.symbol, ours.num_bits, ours.baseline, ref_sym, ref_bits, ref_base
            );
            ll_match = false;
        }
    }
    if ll_match {
        println!("LL table: OK (all 64 states match)");
    }

    // Compare ML table
    println!("\n=== ML Table (64 states) ===");
    let mut ml_match = true;
    #[allow(clippy::needless_range_loop)]
    for state in 0..64 {
        let ours = ml_table.decode(state);
        let (ref_sym, ref_bits, ref_base) = ref_ml_table[state];

        if ours.symbol != ref_sym || ours.num_bits != ref_bits || ours.baseline != ref_base {
            println!(
                "MISMATCH at state {}: ours=({}, {}, {}), ref=({}, {}, {})",
                state, ours.symbol, ours.num_bits, ours.baseline, ref_sym, ref_bits, ref_base
            );
            ml_match = false;
        }
    }
    if ml_match {
        println!("ML table: OK (all 64 states match)");
    }

    // Compare OF table
    println!("\n=== OF Table (32 states) ===");
    let mut of_match = true;
    #[allow(clippy::needless_range_loop)]
    for state in 0..32 {
        let ours = of_table.decode(state);
        let (ref_sym, ref_bits, ref_base) = ref_of_table[state];

        if ours.symbol != ref_sym || ours.num_bits != ref_bits || ours.baseline != ref_base {
            println!(
                "MISMATCH at state {}: ours=({}, {}, {}), ref=({}, {}, {})",
                state, ours.symbol, ours.num_bits, ours.baseline, ref_sym, ref_bits, ref_base
            );
            of_match = false;
        }
    }
    if of_match {
        println!("OF table: OK (all 32 states match)");
    }

    println!("\n=== Summary ===");
    if ll_match && ml_match && of_match {
        println!("All tables match reference zstd!");
    } else {
        println!("Table mismatches found - this explains the decoding failure");
    }
}
