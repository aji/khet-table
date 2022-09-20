use ag::prelude::*;
use autograd as ag;

struct RunningAverage<F> {
    momentum: F,
}

impl<F: ag::Float> ag::op::Op<F> for RunningAverage<F> {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<F>) -> Result<(), ag::op::OpError> {
        let x = ctx.input(0);
        let mut x_avg = ctx.input_mut(1);
        x_avg.zip_mut_with(&x, move |m, x| {
            *m = *m * self.momentum + *x * (F::one() - self.momentum)
        });
        ctx.append_output_view(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<F>) {
        ctx.append_input_grad(Some(ctx.output_grad()));
    }
}

pub fn running_average<'g, F: ag::Float, X, M>(
    g: &'g impl AsGraph<F>,
    x: X,
    x_avg: M,
    momentum: F,
) -> ag::Tensor<'g, F>
where
    X: AsRef<ag::Tensor<'g, F>>,
    M: AsRef<ag::Tensor<'g, F>>,
{
    ag::Tensor::builder(g)
        .append_input(x, false)
        .append_input(x_avg, true)
        .build(RunningAverage { momentum })
}
