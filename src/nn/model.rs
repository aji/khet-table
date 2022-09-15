use ag::ndarray_ext::ArrayRng;
use autograd as ag;

use ag::tensor_ops as T;
use ag::variable::VariableID;
use rand::Rng;

use super::constants::*;
use super::Float;

fn relu_norm_conv_2d<'g, X, W>(
    training: bool,
    x: X,
    w: W,
    pad: usize,
    stride: usize,
) -> ag::Tensor<'g, Float>
where
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    let x = T::conv2d(x, w, pad, stride);
    if training {
        T::normalize(x, &[0, 2, 3])
    } else {
        x
    };
    T::leaky_relu(x, LEAK)
}

// w: [i, j], x: [batches, j], returns [batches, i]
fn linear<'g, W, X>(w: W, x: X) -> ag::Tensor<'g, Float>
where
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    T::matmul(x, T::transpose(w, &[1, 0]))
}

#[derive(Clone)]
pub struct KhetModel {
    input_to_res: InputToRes,
    res_blocks: Vec<ResBlock>,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

#[derive(Clone)]
struct InputToRes {
    conv: VariableID,
}

#[derive(Clone)]
struct ResBlock {
    conv1: VariableID,
    conv2: VariableID,
}

#[derive(Clone)]
struct PolicyHead {
    conv: VariableID,
    fc: VariableID,
}

#[derive(Clone)]
struct ValueHead {
    conv: VariableID,
    fc1: VariableID,
    fc2: VariableID,
}

impl KhetModel {
    pub fn new<'env, 'name, R: Rng>(
        ns: &mut ag::variable::VariableNamespaceMut<'env, 'name, Float>,
        rng: &ArrayRng<Float, R>,
    ) -> KhetModel {
        KhetModel {
            input_to_res: InputToRes {
                conv: ns.slot().name("input_conv").set(rng.random_normal(
                    &[N_FILTERS, N_INPUT_PLANES, 3, 3],
                    0.0,
                    1.0 / N_FILTERS as f64,
                )),
            },

            res_blocks: (0..N_BLOCKS)
                .map(|i| ResBlock {
                    conv1: ns
                        .slot()
                        .name(format!("res{}_conv1", i))
                        .set(rng.random_normal(
                            &[N_FILTERS, N_FILTERS, 3, 3],
                            0.0,
                            1.0 / N_FILTERS as f64,
                        )),
                    conv2: ns
                        .slot()
                        .name(format!("res{}_conv2", i))
                        .set(rng.random_normal(
                            &[N_FILTERS, N_FILTERS, 3, 3],
                            0.0,
                            1.0 / N_FILTERS as f64,
                        )),
                })
                .collect(),

            policy_head: PolicyHead {
                conv: ns
                    .slot()
                    .name("policy_conv")
                    .set(rng.standard_normal(&[2, N_FILTERS, 1, 1])),
                fc: ns
                    .slot()
                    .name("policy_fc")
                    .set(rng.glorot_uniform(&[N_MOVES, 2 * N_ROWS * N_COLS])),
            },

            value_head: ValueHead {
                conv: ns.slot().name("policy_conv").set(rng.random_normal(
                    &[1, N_FILTERS, 1, 1],
                    0.0,
                    1.0 / N_FILTERS as f64,
                )),
                fc1: ns
                    .slot()
                    .name("fc1")
                    .set(rng.glorot_uniform(&[N_VALUE_HIDDEN, N_ROWS * N_COLS])),
                fc2: ns.slot().name("fc2").set(rng.random_uniform(
                    &[1, N_VALUE_HIDDEN],
                    -0.2 / N_VALUE_HIDDEN as f64,
                    0.2 / N_VALUE_HIDDEN as f64,
                )),
            },
        }
    }

    // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
    // out ([batch, N_MOVES], [batch])
    pub fn eval<'env, 'name, 'g>(
        &self,
        training: bool,
        ctx: &'g ag::Context<'env, 'name, Float>,
        input: ag::Tensor<'g, Float>,
    ) -> (ag::Tensor<'g, Float>, ag::Tensor<'g, Float>) {
        let res_input = relu_norm_conv_2d(
            training,
            input,
            ctx.variable_by_id(self.input_to_res.conv),
            1,
            1,
        );

        let tower = {
            let mut res = res_input;

            for block in self.res_blocks.iter() {
                // This is a slight deviation from AlphaZero[1] and AlphaGo
                // Zero[2]. In AGZ they apply the skip connection before the
                // ReLU. Here I'm doing ReLU first before adding it back to the
                // residual stream. In the Architecture section, the AlphaZero
                // paper cites [3] which suggests that putting the ReLU before
                // the residual connection results in better performance, but
                // they also say they used the same architecture as AGZ, which
                // doesn't seem to do this. I'm going to wing it and hope it
                // does better.
                //
                // [1] "A general reinforcement learning algorithm that masters
                // chess, shogi and Go through self-play", D.  Silver, et al
                // https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf
                // [2] "Mastering the game of Go without human knowledge" D.
                // Silver, et al
                // https://www.deepmind.com/blog/alphago-zero-starting-from-scratch
                // [3] "Identity Mappings in Deep Residual Networks", K. He, X.
                // Zhang, S. Ren, J. Sun https://arxiv.org/pdf/1603.05027.pdf

                let x0 = res;
                let x1 = relu_norm_conv_2d(training, x0, ctx.variable_by_id(block.conv1), 1, 1);
                let x2 = relu_norm_conv_2d(training, x1, ctx.variable_by_id(block.conv2), 1, 1);
                res = x0 + x2;
            }

            res
        };

        let policy = {
            let x = relu_norm_conv_2d(
                training,
                tower,
                ctx.variable_by_id(self.policy_head.conv),
                0,
                1,
            );
            let x = T::reshape(x, &[-1, (2 * N_ROWS * N_COLS) as isize]);
            let x = linear(ctx.variable_by_id(self.policy_head.fc), x);
            T::reshape(x, &[-1, N_MOVES as isize])
        };

        let value = {
            let x = relu_norm_conv_2d(
                training,
                tower,
                ctx.variable_by_id(self.value_head.conv),
                0,
                1,
            );
            let x = T::reshape(x, &[-1, (N_ROWS * N_COLS) as isize]);
            let x = T::leaky_relu(linear(ctx.variable_by_id(self.value_head.fc1), x), LEAK);
            let x = T::tanh(linear(ctx.variable_by_id(self.value_head.fc2), x));
            T::reshape(x, &[-1])
        };

        (policy, value)
    }
}
