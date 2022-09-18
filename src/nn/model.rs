use ag::ndarray as nd;
use ag::ndarray::Ix1;
use ag::ndarray_ext::ArrayRng;
use autograd as ag;

use ag::tensor_ops as T;
use ag::variable::VariableID;
use rand::Rng;

use super::constants::*;
use super::Float;

fn relu_norm_conv_2d<'g, X, W, B, Gamma, Beta>(
    x: X,
    w: W,
    b: B,
    bn_gamma: Gamma,
    bn_beta: Beta,
    pad: usize,
    stride: usize,
) -> ag::Tensor<'g, Float>
where
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
    B: AsRef<ag::Tensor<'g, Float>> + Copy,
    Gamma: AsRef<ag::Tensor<'g, Float>> + Copy,
    Beta: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    let x = T::conv2d(x, w, pad, stride) + T::reshape(b, &[1, -1, 1, 1]);
    let x = T::normalize(x, &[0, 2, 3]);
    let x = x * T::reshape(bn_gamma.as_ref(), &[1, -1, 1, 1])
        + T::reshape(bn_beta.as_ref(), &[1, -1, 1, 1]);
    T::leaky_relu(x, LEAK)
}

// w: [i, j], x: [batches, j], returns [batches, i]
fn linear<'g, X, W, B>(x: X, w: W, b: B) -> ag::Tensor<'g, Float>
where
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
    B: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    T::matmul(x, T::transpose(w, &[1, 0])) + T::reshape(b, &[1, -1])
}

#[derive(Clone)]
pub struct KhetModel {
    encoder: Encoder,
    res_blocks: Vec<ResBlock>,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

#[derive(Clone)]
struct Encoder {
    conv: VariableID,
    conv_bias: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
}

#[derive(Clone)]
struct ResBlock {
    conv1: VariableID,
    conv1_bias: VariableID,
    conv1_bn_gamma: VariableID,
    conv1_bn_beta: VariableID,
    conv2: VariableID,
    conv2_bias: VariableID,
    conv2_bn_gamma: VariableID,
    conv2_bn_beta: VariableID,
}

#[derive(Clone)]
struct PolicyHead {
    conv: VariableID,
    conv_bias: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
    fc: VariableID,
    fc_bias: VariableID,
}

#[derive(Clone)]
struct ValueHead {
    conv: VariableID,
    conv_bias: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
    fc1: VariableID,
    fc1_bias: VariableID,
    fc2: VariableID,
    fc2_bias: VariableID,
}

struct ArrayInit<R: Rng> {
    rng: ArrayRng<Float, R>,
}

impl<R: Rng> ArrayInit<R> {
    fn conv_filter(&self, shape: &[usize]) -> nd::Array<Float, nd::IxDyn> {
        self.rng.random_normal(shape, 0.0, 0.01 / shape[0] as f64)
    }
    fn conv_bias(&self, n_channels: usize) -> nd::Array<Float, nd::IxDyn> {
        let f = 0.01 / n_channels as f64;
        self.rng.random_normal(&[n_channels, 1, 1], 0.0, f)
    }
    fn fc_weights(&self, out_features: usize, in_features: usize) -> nd::Array<Float, nd::IxDyn> {
        self.rng.glorot_uniform(&[out_features, in_features])
    }
    fn fc_biases(&self, n_features: usize) -> nd::Array<Float, Ix1> {
        nd::Array::zeros([n_features])
    }
}

impl KhetModel {
    pub fn new<'env, 'name>(
        ns: &mut ag::variable::VariableNamespaceMut<'env, 'name, Float>,
    ) -> KhetModel {
        let init = ArrayInit {
            rng: ArrayRng::default(),
        };

        KhetModel {
            encoder: Encoder {
                conv: ns.slot().name("enc_conv").set(init.conv_filter(&[
                    N_FILTERS,
                    N_INPUT_PLANES,
                    3,
                    3,
                ])),
                conv_bias: ns
                    .slot()
                    .name("enc_conv_bias")
                    .set(init.conv_bias(N_FILTERS)),
                conv_bn_gamma: ns
                    .slot()
                    .name("enc_conv_bn_gamma")
                    .set(init.conv_bias(N_FILTERS)),
                conv_bn_beta: ns
                    .slot()
                    .name("enc_conv_bn_beta")
                    .set(init.conv_bias(N_FILTERS)),
            },

            res_blocks: (0..N_BLOCKS)
                .map(|i| ResBlock {
                    conv1: ns
                        .slot()
                        .name(format!("res{}_conv1", i))
                        .set(init.conv_filter(&[N_FILTERS, N_FILTERS, 3, 3])),
                    conv1_bias: ns
                        .slot()
                        .name(format!("res{}_conv1_bias", i))
                        .set(init.conv_bias(N_FILTERS)),
                    conv1_bn_gamma: ns
                        .slot()
                        .name(format!("res{}_conv1_bn_gamma", i))
                        .set(init.conv_bias(N_FILTERS)),
                    conv1_bn_beta: ns
                        .slot()
                        .name(format!("res{}_conv1_bn_beta", i))
                        .set(init.conv_bias(N_FILTERS)),
                    conv2: ns
                        .slot()
                        .name(format!("res{}_conv2", i))
                        .set(init.conv_filter(&[N_FILTERS, N_FILTERS, 3, 3])),
                    conv2_bias: ns
                        .slot()
                        .name(format!("res{}_conv2_bias", i))
                        .set(init.conv_bias(N_FILTERS)),
                    conv2_bn_gamma: ns
                        .slot()
                        .name(format!("res{}_conv2_bn_gamma", i))
                        .set(init.conv_bias(N_FILTERS)),
                    conv2_bn_beta: ns
                        .slot()
                        .name(format!("res{}_conv2_bn_beta", i))
                        .set(init.conv_bias(N_FILTERS)),
                })
                .collect(),

            policy_head: PolicyHead {
                conv: ns
                    .slot()
                    .name("policy_conv")
                    .set(init.conv_filter(&[2, N_FILTERS, 1, 1])),
                conv_bias: ns.slot().name("policy_conv_bias").set(init.conv_bias(2)),
                conv_bn_gamma: ns
                    .slot()
                    .name("policy_conv_bn_gamma")
                    .set(init.conv_bias(2)),
                conv_bn_beta: ns.slot().name("policy_conv_bn_beta").set(init.conv_bias(2)),
                fc: ns
                    .slot()
                    .name("policy_fc")
                    .set(init.fc_weights(N_MOVES, 2 * N_ROWS * N_COLS)),
                fc_bias: ns
                    .slot()
                    .name("policy_fc_bias")
                    .set(init.fc_biases(N_MOVES)),
            },

            value_head: ValueHead {
                conv: ns
                    .slot()
                    .name("value_conv")
                    .set(init.conv_filter(&[1, N_FILTERS, 1, 1])),
                conv_bias: ns.slot().name("value_conv_bias").set(init.conv_bias(1)),
                conv_bn_gamma: ns.slot().name("value_conv_bn_gamma").set(init.conv_bias(1)),
                conv_bn_beta: ns.slot().name("value_conv_bn_beta").set(init.conv_bias(1)),
                fc1: ns
                    .slot()
                    .name("value_fc1")
                    .set(init.fc_weights(N_VALUE_HIDDEN, N_ROWS * N_COLS)),
                fc1_bias: ns
                    .slot()
                    .name("value_fc1_bias")
                    .set(init.fc_biases(N_VALUE_HIDDEN)),
                fc2: ns
                    .slot()
                    .name("value_fc2")
                    .set(init.fc_weights(1, N_VALUE_HIDDEN)),
                fc2_bias: ns.slot().name("value_fc2_bias").set(init.fc_biases(1)),
            },
        }
    }

    pub fn open<'env, 'name>(
        _ns: &ag::variable::VariableNamespace<'env, 'name, Float>,
    ) -> Result<KhetModel, ()> {
        todo!()
    }

    // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
    // out ([batch, N_MOVES], [batch])
    pub fn eval<'env, 'name, 'g>(
        &self,
        ctx: &'g ag::Context<'env, 'name, Float>,
        input: ag::Tensor<'g, Float>,
    ) -> (ag::Tensor<'g, Float>, ag::Tensor<'g, Float>) {
        let res_input = relu_norm_conv_2d(
            input,
            ctx.variable_by_id(self.encoder.conv),
            ctx.variable_by_id(self.encoder.conv_bias),
            ctx.variable_by_id(self.encoder.conv_bn_gamma),
            ctx.variable_by_id(self.encoder.conv_bn_beta),
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
                let x1 = relu_norm_conv_2d(
                    x0,
                    ctx.variable_by_id(block.conv1),
                    ctx.variable_by_id(block.conv1_bias),
                    ctx.variable_by_id(block.conv1_bn_gamma),
                    ctx.variable_by_id(block.conv1_bn_beta),
                    1,
                    1,
                );
                let x2 = relu_norm_conv_2d(
                    x1,
                    ctx.variable_by_id(block.conv2),
                    ctx.variable_by_id(block.conv2_bias),
                    ctx.variable_by_id(block.conv2_bn_gamma),
                    ctx.variable_by_id(block.conv2_bn_beta),
                    1,
                    1,
                );
                res = x0 + x2;
            }

            res
        };

        let policy = {
            let x = relu_norm_conv_2d(
                tower,
                ctx.variable_by_id(self.policy_head.conv),
                ctx.variable_by_id(self.policy_head.conv_bias),
                ctx.variable_by_id(self.policy_head.conv_bn_gamma),
                ctx.variable_by_id(self.policy_head.conv_bn_beta),
                0,
                1,
            );
            let x = linear(
                T::reshape(x, &[-1, (2 * N_ROWS * N_COLS) as isize]),
                ctx.variable_by_id(self.policy_head.fc),
                ctx.variable_by_id(self.policy_head.fc_bias),
            );
            T::reshape(x, &[-1, N_MOVES as isize])
        };

        let value = {
            let x = relu_norm_conv_2d(
                tower,
                ctx.variable_by_id(self.value_head.conv),
                ctx.variable_by_id(self.value_head.conv_bias),
                ctx.variable_by_id(self.value_head.conv_bn_gamma),
                ctx.variable_by_id(self.value_head.conv_bn_beta),
                0,
                1,
            );
            let x = T::reshape(x, &[-1, (N_ROWS * N_COLS) as isize]);
            let x = T::leaky_relu(
                linear(
                    x,
                    ctx.variable_by_id(self.value_head.fc1),
                    ctx.variable_by_id(self.value_head.fc1_bias),
                ),
                LEAK,
            );
            let x = T::tanh(linear(
                x,
                ctx.variable_by_id(self.value_head.fc2),
                ctx.variable_by_id(self.value_head.fc2_bias),
            ));
            T::reshape(x, &[-1])
        };

        (policy, value)
    }
}
