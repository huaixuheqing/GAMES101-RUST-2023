use nalgebra::{Vector3};
use opencv::core::{MatTraitConst, VecN};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
#[derive(Clone)]
pub struct Texture {
    pub img_data: opencv::core::Mat,
    pub width: usize,
    pub height: usize,
}

impl Texture {
    pub fn new(name: &str) -> Self {
        let img_data = imread(name, IMREAD_COLOR).expect("Image reading error!");
        let width = img_data.cols() as usize;
        let height = img_data.rows() as usize;
        Texture {
            img_data,
            width,
            height,
        }
    }

    pub fn get_color(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        if u < 0.0 { u = 0.0; }
        if u > 1.0 { u = 1.0; }
        if v < 0.0 { v = 0.0; }
        if v > 1.0 { v = 1.0; }

        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let color: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32).unwrap();

        Vector3::new(color[2] as f64, color[1] as f64, color[0] as f64)
    }

    pub fn lerp(x: f64, v_0: Vector3<f64>, v_1: Vector3<f64>) -> Vector3<f64> {
        v_0 + (v_1 - v_0) * x
    }

    pub fn getColorBilinear(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        // 在此实现双线性插值函数, 并替换掉get_color
        if u < 0.0 { u = 0.0; }
        if u > 1.0 { u = 1.0; }
        if v < 0.0 { v = 0.0; }
        if v > 1.0 { v = 1.0; }

        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let color_00: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32).unwrap();
        let color_01: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32 + 1).unwrap();
        let color_10: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32 + 1, u_img as i32).unwrap();
        let color_11: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32 + 1, u_img as i32 + 1).unwrap();

        let color1_00 = Vector3::new(color_00[2] as f64, color_00[1] as f64, color_00[0] as f64);
        let color1_01 = Vector3::new(color_01[2] as f64, color_01[1] as f64, color_01[0] as f64);
        let color1_10 = Vector3::new(color_10[2] as f64, color_10[1] as f64, color_10[0] as f64);
        let color1_11 = Vector3::new(color_11[2] as f64, color_11[1] as f64, color_11[0] as f64);

        let s = v_img - v_img as i32 as f64;
        let u0 = Self::lerp(s,color1_00,color1_10);
        let u1 = Self::lerp(s,color1_01,color1_11);

        let t = u_img - u_img as i32 as f64;
        Self::lerp(t, u0, u1)
    }
}