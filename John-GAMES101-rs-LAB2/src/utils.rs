use std::os::raw::c_void;
use nalgebra::{Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub(crate) fn get_view_matrix(eye_pos: Vector3<f64>) -> Matrix4<f64> {
    let mut view: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    view[(0, 3)] = -eye_pos.x;
    view[(1, 3)] = -eye_pos.y;
    view[(2, 3)] = -eye_pos.z;

    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    let mut model: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    model[(0, 0)] = rotation_angle.to_radians().cos();
    model[(0, 1)] = -rotation_angle.to_radians().sin();
    model[(1, 0)] = rotation_angle.to_radians().sin();
    model[(1, 1)] = rotation_angle.to_radians().cos();

    model
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Matrix4<f64> {
    let mut projection: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    let matrix1 = Matrix4::new(
        z_near,
        0.0,
        0.0,
        0.0,
        0.0,
        z_near,
        0.0,
        0.0,
        0.0,
        0.0,
        z_near + z_far,
        -z_near * z_far,
        0.0,
        0.0,
        1.0,
        0.0,
    );

    let t = - z_near.abs() * (eye_fov.to_radians() / 2.0).tan();
    let r = aspect_ratio * t;
    projection[(0, 0)] = 1.0 / r;
    projection[(1, 1)] = 1.0 / t;
    projection[(2, 2)] = 2.0 / (z_near - z_far).abs();
    let mut projection1 = Matrix4::identity();
    projection1[(2, 3)] = -(z_near + z_far) / 2.0;
    projection * projection1 * matrix1
}


pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<Vector3<f64>>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700, 700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        ).unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image.convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0).expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}