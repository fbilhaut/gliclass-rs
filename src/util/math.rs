use ndarray::*;

pub fn sigmoid<T: NdFloat>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}


pub fn sigmoid_a<T: NdFloat, D: Dimension>(array: &ArrayView<T, D>) -> Array<T, D> {
    array.mapv(sigmoid)
}

