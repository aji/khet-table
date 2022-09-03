pub trait Grad: Sized {
    fn combine(scale: f64, grads: &[Self]) -> Self;
}

impl Grad for Vec<f64> {
    fn combine(scale: f64, grads: &[Vec<f64>]) -> Vec<f64> {
        (0..grads[0].len())
            .map(|i| scale * grads.iter().map(|g| g[i]).sum::<f64>())
            .collect()
    }
}

pub trait Model<Input> {
    type Grad;

    fn forward(&self, features: Input) -> f64;
    fn backward(&self, features: Input, expected: f64) -> (f64, Self::Grad);

    fn apply_grad(&mut self, grad: Self::Grad);
}

pub struct LinearModel {
    w: Vec<f64>,
}

impl LinearModel {
    pub fn new(num_features: usize) -> LinearModel {
        LinearModel {
            w: (0..num_features)
                .map(|_| (rand::random::<f64>() - 0.5) / num_features as f64)
                .collect(),
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.w[..]
    }

    pub fn weights_mut(&mut self) -> &mut [f64] {
        &mut self.w[..]
    }
}

impl<'a> Model<&'a [f64]> for LinearModel {
    type Grad = Vec<f64>;

    fn forward(&self, features: &[f64]) -> f64 {
        let sum: f64 = self.w.iter().zip(features.iter()).map(|(w, x)| w * x).sum();
        sum.tanh()
    }

    fn backward(&self, features: &[f64], expected: f64) -> (f64, Vec<f64>) {
        let s = self.forward(features);
        let loss = (s - expected).powi(2);
        let coeff = 2.0 * (s - expected) * (1.0 - s * s);
        (loss, features.iter().map(|x| coeff * x).collect())
    }

    fn apply_grad(&mut self, grad: Vec<f64>) {
        self.w
            .iter_mut()
            .zip(grad.iter())
            .for_each(|(w, g)| *w -= g);
    }
}
