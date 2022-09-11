use std::marker::PhantomData;

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

/*

n = number of filters
b = number of blocks
v = value hidden layer size

for 18 input planes and 800 moves:

input = n * 18 * 3*3
blocks = b * 2 * n*n * 3*3

policy conv = 2 * n * 3*3
policy fc = 800 * 2*80

value conv = n * 3*3
value fc1 = v * 80
value fc2 = 1 * v

total = n*18*3*3 + b*2*n*n*3*3 + 2*n*3*3 + 800*2*80 + n*3*3 + v*80 + 1*v
      = 162n + 18bnn + 18n + 128000 + 9n + 80v + v
      = 18bnn + 189n + 81v + 128000

for various values of bxn,v:

                   total       in        tower    policy    value
  2x 16, 16 =    141,536 =  2,592 +      9,216 + 128,288 +  1,440
  8x 32,128 =    291,872 =  5,184 +    147,456 + 128,576 + 10,656
 12x 64,128 =  1,035,200 = 10,368 +    884,736 + 129,152 + 10,944
 16x128,256 =  4,891,520 = 20,736 +  4,718,592 + 130,304 + 21,888
 19x256,256 = 22,610,432 = 41,472 + 22,413,312 + 132,608 + 23,040

*/

fn relu_norm_conv_2d<'g, X, W, F: ag::Float>(
    x: X,
    w: W,
    pad: usize,
    stride: usize,
) -> ag::Tensor<'g, F>
where
    X: AsRef<ag::Tensor<'g, F>> + Copy,
    W: AsRef<ag::Tensor<'g, F>> + Copy,
{
    T::relu(T::normalize(T::conv2d(x, w, pad, stride), &[0, 2, 3]))
}

// w: [i, j], x: [batches, j], returns [batches, i]
fn linear<'g, W, X, F: ag::Float>(w: W, x: X) -> ag::Tensor<'g, F>
where
    W: AsRef<ag::Tensor<'g, F>> + Copy,
    X: AsRef<ag::Tensor<'g, F>> + Copy,
{
    T::matmul(x, T::transpose(w, &[1, 0]))
}

struct InputToRes<F> {
    conv: VariableID,
    _phantom: PhantomData<F>,
}

impl<F: ag::Float> InputToRes<F> {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, F>,
        rng: &ArrayRng<F, R>,
        n_filters: usize,
    ) -> InputToRes<F> {
        let shape = &[n_filters, N_INPUT_PLANES, 3, 3];
        InputToRes {
            conv: ns.slot().name("conv").set(rng.standard_normal(shape)),
            _phantom: PhantomData,
        }
    }

    // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
    // out [batch, n_filters, N_ROWS, N_COLS]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, F>,
        x: ag::Tensor<'g, F>,
    ) -> ag::Tensor<'g, F> {
        relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 1, 1)
    }
}

struct ResBlock<F> {
    conv1: VariableID,
    conv2: VariableID,
    _phantom: PhantomData<F>,
}

impl<F: ag::Float> ResBlock<F> {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, F>,
        rng: &ArrayRng<F, R>,
        n_filters: usize,
    ) -> ResBlock<F> {
        let shape = &[n_filters, n_filters, 3, 3];
        ResBlock {
            conv1: ns.slot().name("conv1").set(rng.standard_normal(shape)),
            conv2: ns.slot().name("conv2").set(rng.standard_normal(shape)),
            _phantom: PhantomData,
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch, n_filters, N_ROWS, N_COLS]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, F>,
        x0: ag::Tensor<'g, F>,
    ) -> ag::Tensor<'g, F> {
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

struct PolicyHead<F> {
    conv: VariableID,
    fc: VariableID,
    _phantom: PhantomData<F>,
}

impl<F: ag::Float> PolicyHead<F> {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, F>,
        rng: &ArrayRng<F, R>,
        n_filters: usize,
    ) -> PolicyHead<F> {
        PolicyHead {
            conv: ns
                .slot()
                .name("conv")
                .set(rng.standard_normal(&[2, n_filters, 1, 1])),
            fc: ns
                .slot()
                .name("fc")
                .set(rng.glorot_uniform(&[N_MOVES, 2 * N_ROWS * N_COLS])),
            _phantom: PhantomData,
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch, N_MOVES]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, F>,
        x: &ag::Tensor<'g, F>,
    ) -> ag::Tensor<'g, F> {
        let x = relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 0, 1);
        let x = T::reshape(x, &[-1, (2 * N_ROWS * N_COLS) as isize]);
        let x = linear(ctx.variable_by_id(self.fc), x);
        T::reshape(x, &[-1, N_MOVES as isize])
    }
}

struct ValueHead<F> {
    conv: VariableID,
    fc1: VariableID,
    fc2: VariableID,
    _phantom: PhantomData<F>,
}

impl<F: ag::Float> ValueHead<F> {
    fn new<'env, 'name, R: Rng>(
        mut ns: VariableNamespaceMut<'env, 'name, F>,
        rng: &ArrayRng<F, R>,
        n_filters: usize,
        n_hidden: usize,
    ) -> ValueHead<F> {
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
            _phantom: PhantomData,
        }
    }

    // in [batch, n_filters, N_ROWS, N_COLS]
    // out [batch]
    fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, F>,
        x: &ag::Tensor<'g, F>,
    ) -> ag::Tensor<'g, F> {
        let x = T::relu(T::conv2d(x, ctx.variable_by_id(self.conv), 0, 1));
        let x = T::reshape(x, &[-1, (N_ROWS * N_COLS) as isize]);
        let x = T::relu(linear(ctx.variable_by_id(self.fc1), x));
        let x = T::tanh(linear(ctx.variable_by_id(self.fc2), x));
        T::reshape(x, &[-1])
    }
}

pub struct KhetModel<F> {
    input_to_res: InputToRes<F>,
    res_blocks: Vec<ResBlock<F>>,
    policy_head: PolicyHead<F>,
    value_head: ValueHead<F>,
}

const RES_NAMES: [&'static str; 20] = [
    "res0", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8", "res9", "res10",
    "res11", "res12", "res13", "res14", "res15", "res16", "res17", "res18", "res19",
];

impl<F: ag::Float> KhetModel<F> {
    pub fn new<R: Rng>(
        env: &mut ag::VariableEnvironment<F>,
        rng: &ArrayRng<F, R>,
        n_filters: usize,
        n_blocks: usize,
        n_value_hidden: usize,
    ) -> KhetModel<F> {
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
        ctx: &'g ag::Context<'env, 'name, F>,
        input: ag::Tensor<'g, F>,
    ) -> (ag::Tensor<'g, F>, ag::Tensor<'g, F>) {
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

pub fn default_model<F: ag::Float>(env: &mut ag::VariableEnvironment<F>) -> KhetModel<F> {
    KhetModel::new(
        env,
        &ArrayRng::default(),
        N_FILTERS,
        N_BLOCKS,
        N_VALUE_HIDDEN,
    )
}
