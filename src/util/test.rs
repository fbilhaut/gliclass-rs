use ndarray::*;

/// For testing purposes
pub fn is_close_to<T: NdFloat>(v1: T, v2: T, epsilon: T) -> bool {
    (v1 - v2).abs() <= epsilon
}

/// For testing purposes
pub fn is_close_to_a<T: NdFloat>(a1: &ArrayView1<T>, a2: &[T], epsilon: T) -> bool {
    assert_eq!(a1.len(), a2.len());

    a1.iter()
        .zip(a2.iter())
        .all(|(v1, v2)| is_close_to(*v1, *v2, epsilon))
}
