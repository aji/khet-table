pub type Float = f32;

pub mod model;
pub mod ops;
pub mod search;
pub mod train;

pub mod constants {
    use std::time::Duration;

    pub const N_FILTERS: usize = 20;
    pub const N_BLOCKS: usize = 3;
    pub const N_VALUE_HIDDEN: usize = 64;

    pub const N_MOVES: usize = 800;
    pub const N_ROWS: usize = 8;
    pub const N_COLS: usize = 10;
    pub const N_INPUT_PLANES: usize = 20;

    pub const LEAK: f32 = 0.05;
    pub const LEAK_RES: f32 = 0.0;

    pub const SAMPLING_MOVES: usize = 30;
    pub const DRAW_THRESH: usize = 256;
    pub const ROOT_DIRICHLET_ALPHA: f32 = 0.3;
    pub const ROOT_EXPLORATION_FRACTION: f32 = 0.25;

    pub const PUCB_C_BASE: f32 = 19652.0;
    pub const PUCB_C_INIT: f32 = 1.25;

    pub const BUFFER_SIZE: usize = 2_000_000;
    pub const SELF_PLAY_COST_SCHEDULE: [(usize, usize); 6] = [
        (0, 50),
        (10_000, 100),
        (50_000, 200),
        (100_000, 400),
        (200_000, 600),
        (500_000, 800),
    ];

    pub const BATCH_SIZE: usize = 128;
    pub const WEIGHT_DECAY: f32 = 1e-5;
    pub const GRAD_CLIP: f32 = 1e-2;
    pub const MOMENTUM: f32 = 0.9;
    pub const LR_SCHEDULE: [(usize, f32); 6] = [
        (0, 6e-2),
        (1_000, 2e-2),
        (100_000, 6e-3),
        (300_000, 2e-3),
        (500_000, 6e-4),
        (700_000, 2e-4),
    ];

    pub const BUFFER_PRINT_INTERVAL: Duration = Duration::ZERO;
}

pub use constants::*;
pub use model::KhetModel;
