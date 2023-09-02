use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor};
use crate::{iter_par, run_par};
use burn_tensor::ops::TensorOps;
use burn_tensor::ElementConversion;
use ndarray::{s, ArcArray, Array3, ArrayView3, Axis, CowArray, Ix3, IxDyn};

pub(crate) fn matmul<E, const D: usize>(
    lhs: NdArrayTensor<E, D>,
    rhs: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, D>
where
    E: FloatNdArrayElement,
{
    let shape_ori_lhs = lhs.shape();
    let shape_ori_rhs = rhs.shape();

    let lhs = reshape(&lhs.array);
    let rhs = reshape(&rhs.array);

    let (batch_size_lhs, m, _) = lhs.dim();
    let (batch_size_rhs, _, n) = rhs.dim();

    let mut shape_out = match batch_size_lhs > batch_size_rhs {
        true => shape_ori_lhs,
        false => shape_ori_rhs,
    };
    shape_out.dims[D - 2] = m;
    shape_out.dims[D - 1] = n;

    let out = general_matmul(lhs.view(), rhs.view())
        .into_shape(shape_out.dims.to_vec())
        .unwrap();

    NdArrayTensor::new(out.into_shared())
}

fn general_matmul<E: FloatNdArrayElement>(lhs: ArrayView3<E>, rhs: ArrayView3<E>) -> Array3<E> {
    run_par!(|| {
        let (batch_size_lhs, m, _) = lhs.dim();
        let (batch_size_rhs, k, n) = rhs.dim();
        let max_batch_size = usize::max(batch_size_rhs, batch_size_lhs);

        if batch_size_lhs < max_batch_size && batch_size_lhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        if batch_size_rhs < max_batch_size && batch_size_rhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        let alpha: E = 1.0.elem();
        let beta: E = 0.0.elem();

        let mut out_array = ndarray::Array3::<E>::zeros((max_batch_size, m, n));

        let lhs_array = lhs.to_shape((batch_size_lhs, m, k)).unwrap();
        let rhs_array = rhs.to_shape((batch_size_rhs, k, n)).unwrap();

        iter_par!(out_array.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(b, mut out)| {
                let lhs_slice = match batch_size_lhs == 1 {
                    true => lhs_array.slice(s!(0, .., ..)),
                    false => lhs_array.slice(s!(b, .., ..)),
                };
                let rhs_slice = match batch_size_rhs == 1 {
                    true => rhs_array.slice(s!(0, .., ..)),
                    false => rhs_array.slice(s!(b, .., ..)),
                };

                ndarray::linalg::general_mat_mul(alpha, &lhs_slice, &rhs_slice, beta, &mut out);
            });

        out_array
    })
}

fn reshape<E: FloatNdArrayElement>(array: &'_ ArcArray<E, IxDyn>) -> CowArray<'_, E, Ix3> {
    let shape = array.shape();
    let len = shape.len();

    let target_shape = [
        (&shape[..len - 2]).iter().fold(1, |acc, e| acc * e),
        shape[len - 2],
        shape[len - 1],
    ];

    array.to_shape(target_shape).unwrap()
}
