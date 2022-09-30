pub type Float = f32;

pub mod model;
pub mod ops;
pub mod search;
pub mod train;

pub mod constants {
    use std::time::Duration;

    pub const NS_KHET: &'static str = "khet";
    pub const NS_OPT: &'static str = "opt";

    pub const N_FILTERS: usize = 36;
    pub const N_BLOCKS: usize = 4;
    pub const N_VALUE_HIDDEN: usize = 256;

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
    pub const TRAIN_TEST_RATIO: usize = 10;
    pub const SELF_PLAY_COST_SCHEDULE: [(usize, usize); 6] = [
        (0, 50),
        (10_000, 100),
        (50_000, 200),
        (100_000, 400),
        (200_000, 600),
        (500_000, 800),
    ];

    pub const BATCH_SIZE: usize = 32;
    pub const WEIGHT_DECAY: f32 = 1e-4;
    pub const GRAD_CLIP: f32 = 1.0;
    pub const GRAD_CLIP_L2_NORM: bool = true;
    pub const MOMENTUM: f32 = 0.7;
    pub const LR_SCHEDULE: [(usize, f32); 4] =
        [(0, 2e-1), (10_000, 2e-2), (100_000, 2e-3), (500_000, 2e-4)];

    pub const BUFFER_PRINT_INTERVAL: Duration = Duration::from_secs(1);
    pub const TRAIN_PRINT_INTERVAL: Duration = Duration::from_secs(1);
}

pub use constants::*;
pub use model::KhetModel;
