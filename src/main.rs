#[link(name = "cc_main")]
extern "C" {
    fn cc_main() -> i32;
}

fn main() {
    unsafe {
        cc_main();
    }
}
