use ndarray::*;

/// For testing purposes
pub fn is_close_to<T: NdFloat>(v1: T, v2: T, epsilon: T) -> bool {
    (v1 - v2).abs() <= epsilon
}

/// For testing purposes
pub fn is_close_to_a<T: NdFloat>(a1: &ArrayView1<T>, a2: &[T], epsilon: T) -> bool {
    assert!(a1.len() == a2.len());
    for i in 0..a1.len() {
        if !is_close_to(*a1.get(i).unwrap(), a2[i], epsilon) {
            return false;
        }
    }
    true
}
