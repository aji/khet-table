use ag::ndarray as nd;
use ag::ndarray_ext::ArrayRng;
use ag::prelude::*;
use autograd as ag;

use ag::tensor_ops as T;
use ag::variable::VariableID;
use rand::Rng;

use super::constants::*;
use super::ops;
use super::Float;

fn relu_norm_conv_2d<'g, X, W, B, BnMean, BnStdDev, BnGamma, BnBeta>(
    g: &'g impl AsGraph<Float>,
    x: X,
    w: W,
    b: B,
    bn_mean: BnMean,
    bn_stddev: BnStdDev,
    bn_gamma: BnGamma,
    bn_beta: BnBeta,
    pad: usize,
    stride: usize,
    inference: bool,
) -> ag::Tensor<'g, Float>
where
    X: AsRef<ag::Tensor<'g, Float>> + Copy,
    W: AsRef<ag::Tensor<'g, Float>> + Copy,
    B: AsRef<ag::Tensor<'g, Float>> + Copy,
    BnMean: AsRef<ag::Tensor<'g, Float>> + Copy,
    BnStdDev: AsRef<ag::Tensor<'g, Float>> + Copy,
    BnGamma: AsRef<ag::Tensor<'g, Float>> + Copy,
    BnBeta: AsRef<ag::Tensor<'g, Float>> + Copy,
{
    // conv2d with bias
    let x = T::conv2d(x, w, pad, stride) + T::reshape(b, &[1, -1, 1, 1]);

    // batch normalization
    let x = if inference {
        (x - T::reshape(bn_mean.as_ref(), &[1, -1, 1, 1]))
            * T::reshape(T::inv_sqrt(bn_stddev.as_ref() + 1e-5), &[1, -1, 1, 1])
    } else {
        let axes = &[0, 2, 3];
        let mean = ops::running_average(
            g,
            T::reshape(T::reduce_mean(x, axes, true), &[-1, 1, 1]),
            bn_mean.as_ref(),
            0.9,
        );
        let centered = x - T::reshape(mean, &[1, -1, 1, 1]);
        let variance = ops::running_average(
            g,
            T::reshape(T::reduce_mean(T::square(centered), axes, true), &[-1, 1, 1]),
            bn_stddev.as_ref(),
            0.9,
        );
        centered * T::inv_sqrt(T::reshape(variance, &[1, -1, 1, 1]) + 1e-5)
    };
    let x = x * T::reshape(bn_gamma.as_ref(), &[1, -1, 1, 1])
        + T::reshape(bn_beta.as_ref(), &[1, -1, 1, 1]);

    // relu
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
    conv_bn_mean: VariableID,
    conv_bn_stddev: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
}

#[derive(Clone)]
struct ResBlock {
    conv1: VariableID,
    conv1_bias: VariableID,
    conv1_bn_mean: VariableID,
    conv1_bn_stddev: VariableID,
    conv1_bn_gamma: VariableID,
    conv1_bn_beta: VariableID,
    conv2: VariableID,
    conv2_bias: VariableID,
    conv2_bn_mean: VariableID,
    conv2_bn_stddev: VariableID,
    conv2_bn_gamma: VariableID,
    conv2_bn_beta: VariableID,
}

#[derive(Clone)]
struct PolicyHead {
    conv: VariableID,
    conv_bias: VariableID,
    conv_bn_mean: VariableID,
    conv_bn_stddev: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
    fc: VariableID,
    fc_bias: VariableID,
}

#[derive(Clone)]
struct ValueHead {
    conv: VariableID,
    conv_bias: VariableID,
    conv_bn_mean: VariableID,
    conv_bn_stddev: VariableID,
    conv_bn_gamma: VariableID,
    conv_bn_beta: VariableID,
    fc1: VariableID,
    fc1_bias: VariableID,
    fc2: VariableID,
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
    fn fc_biases(&self, n_features: usize) -> nd::Array<Float, nd::Ix1> {
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
                conv_bn_mean: ns
                    .slot()
                    .name("enc_conv_bn_mean")
                    .set(nd::Array::zeros([N_FILTERS, 1, 1])),
                conv_bn_stddev: ns
                    .slot()
                    .name("enc_conv_bn_stddev")
                    .set(nd::Array::ones([N_FILTERS, 1, 1])),
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
                    conv1_bn_mean: ns
                        .slot()
                        .name(format!("res{}_conv1_bn_mean", i))
                        .set(nd::Array::zeros([N_FILTERS, 1, 1])),
                    conv1_bn_stddev: ns
                        .slot()
                        .name(format!("res{}_conv1_bn_stddev", i))
                        .set(nd::Array::ones([N_FILTERS, 1, 1])),
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
                    conv2_bn_mean: ns
                        .slot()
                        .name(format!("res{}_conv2_bn_mean", i))
                        .set(nd::Array::zeros([N_FILTERS, 1, 1])),
                    conv2_bn_stddev: ns
                        .slot()
                        .name(format!("res{}_conv2_bn_stddev", i))
                        .set(nd::Array::ones([N_FILTERS, 1, 1])),
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
                conv_bn_mean: ns
                    .slot()
                    .name("policy_conv_bn_mean")
                    .set(nd::Array::zeros([2, 1, 1])),
                conv_bn_stddev: ns
                    .slot()
                    .name("policy_conv_bn_stddev")
                    .set(nd::Array::ones([2, 1, 1])),
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
                conv_bn_mean: ns
                    .slot()
                    .name("value_conv_bn_mean")
                    .set(nd::Array::zeros([1, 1, 1])),
                conv_bn_stddev: ns
                    .slot()
                    .name("value_conv_bn_stddev")
                    .set(nd::Array::ones([1, 1, 1])),
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
                    .set(nd::Array::zeros([1, N_VALUE_HIDDEN])),
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
        inference: bool,
    ) -> (ag::Tensor<'g, Float>, ag::Tensor<'g, Float>) {
        let res_input = relu_norm_conv_2d(
            ctx,
            input,
            ctx.variable_by_id(self.encoder.conv),
            ctx.variable_by_id(self.encoder.conv_bias),
            ctx.variable_by_id(self.encoder.conv_bn_mean),
            ctx.variable_by_id(self.encoder.conv_bn_stddev),
            ctx.variable_by_id(self.encoder.conv_bn_gamma),
            ctx.variable_by_id(self.encoder.conv_bn_beta),
            1,
            1,
            inference,
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
                    ctx,
                    x0,
                    ctx.variable_by_id(block.conv1),
                    ctx.variable_by_id(block.conv1_bias),
                    ctx.variable_by_id(block.conv1_bn_mean),
                    ctx.variable_by_id(block.conv1_bn_stddev),
                    ctx.variable_by_id(block.conv1_bn_gamma),
                    ctx.variable_by_id(block.conv1_bn_beta),
                    1,
                    1,
                    inference,
                );
                let x2 = relu_norm_conv_2d(
                    ctx,
                    x1,
                    ctx.variable_by_id(block.conv2),
                    ctx.variable_by_id(block.conv2_bias),
                    ctx.variable_by_id(block.conv2_bn_mean),
                    ctx.variable_by_id(block.conv2_bn_stddev),
                    ctx.variable_by_id(block.conv2_bn_gamma),
                    ctx.variable_by_id(block.conv2_bn_beta),
                    1,
                    1,
                    inference,
                );
                res = x0 + x1 * LEAK_RES + x2;
            }

            res
        };

        let policy = {
            let x = relu_norm_conv_2d(
                ctx,
                tower,
                ctx.variable_by_id(self.policy_head.conv),
                ctx.variable_by_id(self.policy_head.conv_bias),
                ctx.variable_by_id(self.policy_head.conv_bn_mean),
                ctx.variable_by_id(self.policy_head.conv_bn_stddev),
                ctx.variable_by_id(self.policy_head.conv_bn_gamma),
                ctx.variable_by_id(self.policy_head.conv_bn_beta),
                0,
                1,
                inference,
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
                ctx,
                tower,
                ctx.variable_by_id(self.value_head.conv),
                ctx.variable_by_id(self.value_head.conv_bias),
                ctx.variable_by_id(self.value_head.conv_bn_mean),
                ctx.variable_by_id(self.value_head.conv_bn_stddev),
                ctx.variable_by_id(self.value_head.conv_bn_gamma),
                ctx.variable_by_id(self.value_head.conv_bn_beta),
                0,
                1,
                inference,
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
                T::zeros(&[1], ctx),
            ));
            T::reshape(x, &[-1])
        };

        (policy, value)
    }
}
