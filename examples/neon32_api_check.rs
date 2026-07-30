#[cfg(all(target_arch = "arm", feature = "arm-neon"))]
fn f(n: pulp::arm::Neon) {
    use pulp::f32x4;
    let a = n.splat_f32x4(1.0);
    let b = n.splat_f32x4(2.0);
    let _c: f32x4 = n.mul_f32x4(a, b);
    let _d: f32x4 = n.add_f32x4(a, b);
    let _e: f32x4 = n.sub_f32x4(a, b);
}
fn main() {}
