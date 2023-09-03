use burn_tensor::{Data, Shape};

use ndarray::{ArcArray, Array, Dim, IxDyn};

#[derive(new, Debug, Clone)]
pub struct NdArrayTensor<E, const D: usize> {
    pub array: ArcArray<E, IxDyn>,
}

impl<E, const D: usize> NdArrayTensor<E, D> {
    pub(crate) fn shape(&self) -> Shape<D> {
        Shape::from(self.array.shape().to_vec())
    }
}

#[cfg(test)]
mod utils {
    use super::*;
    use crate::element::FloatNdArrayElement;

    impl<E, const D: usize> NdArrayTensor<E, D>
    where
        E: Default + Clone,
    {
        pub(crate) fn into_data(self) -> Data<E, D>
        where
            E: FloatNdArrayElement,
        {
            let shape = self.shape();
            let values = self.array.into_iter().collect();

            Data::new(values, shape)
        }
    }
}

impl<E, const D: usize> NdArrayTensor<E, D>
where
    E: Default + Clone,
{
    pub fn from_data(data: Data<E, D>) -> NdArrayTensor<E, D> {
        let shape = data.shape.clone();
        let array = Array::from_vec(data.value).into_shared();

        NdArrayTensor {
            array: array.into_shape(shape.dims.to_vec()).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::Distribution;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(
            Shape::new([3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(
            Shape::new([2, 3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = Data::<f32, 3>::random(
            Shape::new([2, 3, 4]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = Data::<f32, 4>::random(
            Shape::new([2, 3, 4, 2]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
