use ag::ndarray_ext::ArrayRng;
use autograd as ag;

use ag::tensor_ops as T;
use ag::variable::{VariableID, VariableNamespaceMut};
use rand::Rng;

const N_FILTERS: usize = 32;
const N_BLOCKS: usize = 8;
const N_VALUE_HIDDEN: usize = 256;

const N_MOVES: usize = 800;
const N_ROWS: usize = 8;
const N_COLS: usize = 10;
const N_INPUT_PLANES: usize = 20;

pub type Float = f32;

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

fn relu_norm_conv_2d<'g, X, W>(x: X, w: W, pad: usize, stride: usize) -> ag::Tensor<'g, Float>
where
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    T::relu(T::normalize(T::conv2d(x, w, pad, stride), &[0, 2, 3]))
}

// w: [i, j], x: [batches, j], returns [batches, i]
fn linear<'g, W, X>(w: W, x: X) -> ag::Tensor<'g, Float>
where
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    T::matmul(x, T::transpose(w, &[1, 0]))
}

struct InputToRes {
    conv: VariableID,
}

impl InputToRes {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, Float>,
        rng: &ArrayRng<Float, R>,
        n_filters: usize,
    ) -> InputToRes {
        let shape = &[n_filters, N_INPUT_PLANES, 3, 3];
        InputToRes {
            conv: ns.slot().name("conv").set(rng.standard_normal(shape)),
        }
    }

    // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
    // out [batch, n_filters, N_ROWS, N_COLS]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        x: ag::Tensor<'g, Float>,
    ) -> ag::Tensor<'g, Float> {
        relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 1, 1)
    }
}

struct ResBlock {
    conv1: VariableID,
    conv2: VariableID,
}

impl ResBlock {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, Float>,
        rng: &ArrayRng<Float, R>,
        n_filters: usize,
    ) -> ResBlock {
        let shape = &[n_filters, n_filters, 3, 3];
        ResBlock {
            conv1: ns.slot().name("conv1").set(rng.standard_normal(shape)),
            conv2: ns.slot().name("conv2").set(rng.standard_normal(shape)),
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch, n_filters, N_ROWS, N_COLS]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        x0: ag::Tensor<'g, Float>,
    ) -> ag::Tensor<'g, Float> {
        // This is a slight deviation from AlphaZero[1] and AlphaGo Zero[2]. In
        // AGZ they apply the skip connection before the ReLU. Here I'm doing
        // ReLU first before adding it back to the residual stream. In the
        // Architecture section, the AlphaZero paper cites [3] which suggests
        // that putting the ReLU before the residual connection results in
        // better performance, but they also say they used the same architecture
        // as AGZ, which doesn't seem to do this. I'm going to wing it and hope
        // it does better.
        //
        // [1] "A general reinforcement learning algorithm that masters chess,
        // shogi and Go through self-play", D. Silver, et al
        // https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf
        // [2] "Mastering the game of Go without human knowledge" D. Silver, et
        // al https://www.deepmind.com/blog/alphago-zero-starting-from-scratch
        // [3] "Identity Mappings in Deep Residual Networks", K. He, X. Zhang,
        // S. Ren, J. Sun https://arxiv.org/pdf/1603.05027.pdf

        let x1 = relu_norm_conv_2d(x0, ctx.variable_by_id(self.conv1), 1, 1);
        let x2 = relu_norm_conv_2d(x1, ctx.variable_by_id(self.conv2), 1, 1);
        x0 + x2
    }
}

struct PolicyHead {
    conv: VariableID,
    fc: VariableID,
}

impl PolicyHead {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, Float>,
        rng: &ArrayRng<Float, R>,
        n_filters: usize,
    ) -> PolicyHead {
        PolicyHead {
            conv: ns
                .slot()
                .name("conv")
                .set(rng.standard_normal(&[2, n_filters, 1, 1])),
            fc: ns
                .slot()
                .name("fc")
                .set(rng.glorot_uniform(&[N_MOVES, 2 * N_ROWS * N_COLS])),
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch, N_MOVES]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        x: &ag::Tensor<'g, Float>,
    ) -> ag::Tensor<'g, Float> {
        let x = relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 0, 1);
        let x = T::reshape(x, &[-1, (2 * N_ROWS * N_COLS) as isize]);
        let x = linear(ctx.variable_by_id(self.fc), x);
        T::reshape(x, &[-1, N_MOVES as isize])
    }
}

struct ValueHead {
    conv: VariableID,
    fc1: VariableID,
    fc2: VariableID,
}

impl ValueHead {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, Float>,
        rng: &ArrayRng<Float, R>,
        n_filters: usize,
        n_hidden: usize,
    ) -> ValueHead {
        ValueHead {
            conv: ns
                .slot()
                .name("conv")
                .set(rng.standard_normal(&[1, n_filters, 1, 1])),
            fc1: ns
                .slot()
                .name("fc1")
                .set(rng.glorot_uniform(&[n_hidden, N_ROWS * N_COLS])),
            fc2: ns.slot().name("fc2").set(rng.random_uniform(
                &[1, n_hidden],
                -1.0 / n_hidden as f64,
                1.0 / n_hidden as f64,
            )),
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        x: &ag::Tensor<'g, Float>,
    ) -> ag::Tensor<'g, Float> {
        let x = T::relu(T::conv2d(x, ctx.variable_by_id(self.conv), 0, 1));
        let x = T::reshape(x, &[-1, (N_ROWS * N_COLS) as isize]);
        let x = T::relu(linear(ctx.variable_by_id(self.fc1), x));
        let x = T::tanh(linear(ctx.variable_by_id(self.fc2), x));
        T::reshape(x, &[-1])
    }
}

pub struct KhetModel {
    input_to_res: InputToRes,
    res_blocks: Vec<ResBlock>,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

const RES_NAMES: [&'static str; 20] = [
    "res0", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8", "res9", "res10",
    "res11", "res12", "res13", "res14", "res15", "res16", "res17", "res18", "res19",
];

impl KhetModel {
    pub fn new<R: Rng>(
        env: &mut ag::VariableEnvironment<Float>,
        rng: &ArrayRng<Float, R>,
        n_filters: usize,
        n_blocks: usize,
        n_value_hidden: usize,
    ) -> KhetModel {
        KhetModel {
            input_to_res: InputToRes::new(env.namespace_mut("input"), rng, n_filters),
            res_blocks: (0..n_blocks)
                .map(|i| ResBlock::new(env.namespace_mut(RES_NAMES[i]), rng, n_filters))
                .collect(),
            policy_head: PolicyHead::new(env.namespace_mut("policy"), rng, n_filters),
            value_head: ValueHead::new(env.namespace_mut("value"), rng, n_filters, n_value_hidden),
        }
    }

    // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
    // out ([batch, N_MOVES], [batch])
    pub fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        input: ag::Tensor<'g, Float>,
    ) -> (ag::Tensor<'g, Float>, ag::Tensor<'g, Float>) {
        let res = {
            let mut res = self.input_to_res.eval(ctx, input);
            for block in self.res_blocks.iter() {
                res = block.eval(ctx, res);
            }
            res
        };

        let policy = self.policy_head.eval(ctx, &res);
        let value = self.value_head.eval(ctx, &res);

        (policy, value)
    }
}

pub fn default_model(env: &mut ag::VariableEnvironment<Float>) -> KhetModel {
    KhetModel::new(
        env,
        &ArrayRng::default(),
        N_FILTERS,
        N_BLOCKS,
        N_VALUE_HIDDEN,
    )
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};

    use super::*;
    use crate::bb;

    #[bench]
    fn bench_nn_forward(b: &mut Bencher) {
        let (env, model) = {
            let mut env = ag::VariableEnvironment::<Float>::new();
            let model = default_model(&mut env);
            (env, model)
        };

        env.run(|g| {
            let img = g.placeholder("img", &[20, 8, 10]);
            let x = T::reshape(img, &[1, 20, 8, 10]);
            let (policy, value) = model.eval(g, x);
            let top = T::concat(&[policy, T::reshape(value, &[-1, 1])], 1);

            b.iter(|| {
                let board = bb::Board::new_classic();
                black_box(
                    g.evaluator()
                        .push(top)
                        .feed("img", board.nn_image().view())
                        .run(),
                );
            });
        });
    }
}
