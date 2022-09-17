pub type Float = f32;

pub mod model;
pub mod search;
pub mod train;

pub mod constants {
    pub const N_FILTERS: usize = 36;
    pub const N_BLOCKS: usize = 3;
    pub const N_VALUE_HIDDEN: usize = 256;

    pub const N_MOVES: usize = 800;
    pub const N_ROWS: usize = 8;
    pub const N_COLS: usize = 10;
    pub const N_INPUT_PLANES: usize = 20;

    pub const LEAK: f32 = 0.05;

    pub const SAMPLING_MOVES: usize = 30;
    pub const DRAW_THRESH: usize = 256;
    pub const TRAIN_ITERS: usize = 800;
    pub const ROOT_DIRICHLET_ALPHA: f32 = 0.3;
    pub const ROOT_EXPLORATION_FRACTION: f32 = 0.25;

    pub const PUCB_C_BASE: f32 = 19652.0;
    pub const PUCB_C_INIT: f32 = 1.25;

    pub const TRAIN_STEPS: usize = 700_000;
    pub const BUFFER_SIZE: usize = 1_000_000;
    pub const BATCH_SIZE: usize = 1024;

    pub const WEIGHT_DECAY: f32 = 1e-4;
    pub const GRAD_CLIP: f32 = 1.0;
    pub const MOMENTUM: f32 = 0.9;
    pub const LR_SCHEDULE: [(usize, f32); 4] =
        [(0, 2e-1), (100_000, 2e-2), (300_000, 2e-3), (500_000, 2e-4)];
}

pub use constants::*;
pub use model::KhetModel;

/*

n = number of filters
b = number of blocks
v = value hidden layer size

for 20 input planes and 800 moves:

input = n * 20 * 3*3
blocks = b * 2 * n*n * 3*3

policy conv = 2 * n * 1*1
policy fc = 800 * 2*80

value conv = n * 1*1
value fc1 = v * 80
value fc2 = 1 * v

total = n*20*3*3 + b*2*n*n*3*3 + 2*n*1*1 + 800*2*80 + n*1*1 + v*80 + 1*v
      = 180n + 18bnn + 2n + 128000 + n + 80v + v
      = 18bnn + 183n + 81v + 128000

for various values of bxn,v:

                   total       in        tower    policy    value
  2x 16, 16 =    141,440 =  2,880 +      9,216 + 128,032 +  1,312
  8x 32,128 =    291,680 =  5,760 +    147,456 + 128,064 + 10,400
 12x 64,128 =  1,034,816 = 11,520 +    884,736 + 128,128 + 10,432
 16x128,256 =  4,890,752 = 23,040 +  4,718,592 + 128,256 + 20,864
 19x256,256 = 22,608,896 = 46,080 + 22,413,312 + 128,512 + 20,992

*/
