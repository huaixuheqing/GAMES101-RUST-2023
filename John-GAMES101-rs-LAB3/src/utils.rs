use std::os::raw::c_void;
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use opencv::core::{Mat, MatTraitConst, sqrt};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};
use crate::shader::{FragmentShaderPayload, VertexShaderPayload};
use crate::texture::Texture;
use crate::triangle::Triangle;

type V3f = Vector3<f64>;
type M4f = Matrix4<f64>;

pub(crate) fn get_view_matrix(eye_pos: V3f) -> M4f {
    let mut view: M4f = Matrix4::identity();
    view[(0, 3)] = -eye_pos[0];
    view[(1, 3)] = -eye_pos[1];
    view[(2, 3)] = -eye_pos[2];

    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> M4f {
    let mut model: M4f = Matrix4::identity();
    let rad = rotation_angle.to_radians();
    model[(0, 0)] = rad.cos();
    model[(2, 2)] = model[(0, 0)];
    model[(0, 2)] = rad.sin();
    model[(2, 0)] = -model[(0, 2)];
    let mut scale: M4f = Matrix4::identity();
    scale[(0, 0)] = 2.5;
    scale[(1, 1)] = 2.5;
    scale[(2, 2)] = 2.5;
    model * scale
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> M4f {
    let mut persp2ortho: M4f = Matrix4::zeros();
    /*  Implement your code here  */

    let t = -z_near * (eye_fov.to_radians() / 2.0).tan();
    let r = t * aspect_ratio;
    let l = -r;
    let b = -t;

    let mut persp = Matrix4::identity();
    persp[(0,0)] = z_near;
    persp[(1,1)] = z_near;
    persp[(2,2)] = z_near + z_far;
    persp[(2,3)] = -z_far * z_near;
    persp[(3,2)] = 1.0;
    persp[(3,3)] = 0.0;

    let mut ortho1 = Matrix4::identity();
    ortho1[(0,0)] = 1.0 / r;
    ortho1[(1,1)] = 1.0 / t;
    ortho1[(2,2)] = 2.0 / (z_near - z_far);

    let mut ortho2 = Matrix4::identity();
    ortho2[(0, 3)] = (r + l) / -2.0;
    ortho2[(1, 3)] = (t + b) / -2.0;
    ortho2[(2, 3)] = (z_far + z_near) / -2.0;

    persp2ortho = ortho1 * ortho2 * persp;

    persp2ortho
}


pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<V3f>) -> Mat {
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

pub fn load_triangles(obj_file: &str) -> Vec<Triangle> {
    let (models, _) = tobj::load_obj(&obj_file, &tobj::LoadOptions::default()).unwrap();
    let mesh = &models[0].mesh;
    let n = mesh.indices.len() / 3;
    let mut triangles = vec![Triangle::default(); n];

    // 遍历模型的每个面
    for vtx in 0..n {
        let rg = vtx * 3..vtx * 3 + 3;
        let idx: Vec<_> = mesh.indices[rg.clone()].iter().map(|i| *i as usize).collect();

        // 记录图形每个面中连续三个顶点（小三角形）
        for j in 0..3 {
            let v = &mesh.positions[3 * idx[j]..3 * idx[j] + 3];
            triangles[vtx].set_vertex(j, Vector4::new(v[0] as f64, v[1] as f64, v[2] as f64, 1.0));
            let ns = &mesh.normals[3 * idx[j]..3 * idx[j] + 3];
            triangles[vtx].set_normal(j, Vector3::new(ns[0] as f64, ns[1] as f64, ns[2] as f64));
            let tex = &mesh.texcoords[2 * idx[j]..2 * idx[j] + 2];
            triangles[vtx].set_tex_coord(j, tex[0] as f64, tex[1] as f64);
        }
    }
    triangles
}

// 选择对应的Shader
pub fn choose_shader_texture(method: &str,
                             obj_path: &str) -> (fn(&FragmentShaderPayload) -> Vector3<f64>, Option<Texture>) {
    let mut active_shader: fn(&FragmentShaderPayload) -> Vector3<f64> = phong_fragment_shader;
    let mut tex = None;
    if method == "normal" {
        println!("Rasterizing using the normal shader");
        active_shader = normal_fragment_shader;
    } else if method == "texture" {
        println!("Rasterizing using the normal shader");
        active_shader = texture_fragment_shader;
        tex = Some(Texture::new(&(obj_path.to_owned() + "spot_texture.png")));
    } else if method == "phong" {
        println!("Rasterizing using the phong shader");
        active_shader = phong_fragment_shader;
    } else if method == "bump" {
        println!("Rasterizing using the bump shader");
        active_shader = bump_fragment_shader;
    } else if method == "displacement" {
        println!("Rasterizing using the displacement shader");
        active_shader = displacement_fragment_shader;
    }
    (active_shader, tex)
}

pub fn vertex_shader(payload: &VertexShaderPayload) -> V3f {
    payload.position
}

#[derive(Default)]
struct Light {
    pub position: V3f,
    pub intensity: V3f,
}

pub fn normal_fragment_shader(payload: &FragmentShaderPayload) -> V3f {
    let result_color =
        (payload.normal.xyz().normalize() + Vector3::new(1.0, 1.0, 1.0)) / 2.0;
    result_color * 255.0
}

pub fn length_squared(a: Vector3<f64>) -> f64 {
    (a.x * a.x + a.y * a.y + a.z * a.z)
}

pub fn length(a: Vector3<f64>) -> f64 {
    (a.x * a.x + a.y * a.y + a.z * a.z).sqrt()
}

pub fn elem_mul(a: Vector3<f64>, b: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

pub fn phong_fragment_shader(payload: &FragmentShaderPayload) -> V3f {
    // 泛光、漫反射、高光系数
    let ka = Vector3::new(0.005, 0.005, 0.005);
    let kd = payload.color;
    let ks = Vector3::new(0.7937, 0.7937, 0.7937);

    // 灯光位置和强度
    let l1 = Light {
        position: Vector3::new(20.0, 20.0, 20.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let l2 = Light {
        position: Vector3::new(-20.0, 20.0, 0.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let lights = vec![l1, l2];
    let amb_light_intensity = Vector3::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::new(0.0, 0.0, 10.0);

    let p = 150.0;

    // ping point的信息
    let normal = payload.normal;
    let point = payload.view_pos;
    let color = payload.color;

    let mut result_color = Vector3::zeros(); // 保存光照结果

    // <遍历每一束光>
    for light in lights {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.

        let l = (light.position - point) / length(light.position - point);
        let v = (eye_pos - point) / length(eye_pos - point);
        let h = (l + v) / length(l + v);
        let n = normal / length(normal);

        let cos_theta = l.dot(&n);
        let diffusely_reflected_light = elem_mul(kd, light.intensity / length_squared(light.position - point)) * cos_theta.max(0.0);
        result_color += diffusely_reflected_light;
        let cos_alpha = h.dot(&n);
        let specularly_reflected_light = elem_mul(ks, light.intensity / length_squared(light.position - point)) * cos_alpha.max(0.0).powf(p);
        result_color += specularly_reflected_light;
        let reflected_ambient_light = elem_mul(ka,amb_light_intensity);
        result_color += reflected_ambient_light;
    }

    result_color * 255.0
}

pub fn texture_fragment_shader(payload: &FragmentShaderPayload) -> V3f {
    let ka = Vector3::new(0.005, 0.005, 0.005);
    let texture_color: Vector3<f64> = match &payload.texture {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        // <获取材质颜色信息>
        None => Vector3::new(0.0, 0.0, 0.0),
        Some(texture) => texture.getColorBilinear(payload.tex_coords.x,payload.tex_coords.y), // Do modification here
    };
    let kd = texture_color / 255.0; // 材质颜色影响漫反射系数
    let ks = Vector3::new(0.7937, 0.7937, 0.7937);

    let l1 = Light {
        position: Vector3::new(20.0, 20.0, 20.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let l2 = Light {
        position: Vector3::new(-20.0, 20.0, 0.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let lights = vec![l1, l2];
    let amb_light_intensity = Vector3::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::new(0.0, 0.0, 10.0);

    let p = 150.0;

    let color = texture_color;
    let point = payload.view_pos;
    let normal = payload.normal;
    let mut result_color = Vector3::zeros();

    for light in lights {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.
        let l = (light.position - point).normalize();
        let v = (eye_pos - point).normalize();
        let h = (l + v).normalize();
        let n = normal.normalize();

        let cos_theta = l.dot(&n);
        let diffusely_reflected_light = elem_mul(kd, light.intensity / length_squared(light.position - point)) * cos_theta.max(0.0);
        result_color += diffusely_reflected_light;
        let cos_alpha = h.dot(&n);
        let specularly_reflected_light = elem_mul(ks, light.intensity / length_squared(light.position - point)) * cos_alpha.max(0.0).powf(p);
        result_color += specularly_reflected_light;
        let reflected_ambient_light = elem_mul(ka,amb_light_intensity);
        result_color += reflected_ambient_light;
    }


    result_color * 255.0
}

pub fn h(payload: &FragmentShaderPayload, u: f64, v: f64) -> f64 {
    let texture_color: Vector3<f64> = match &payload.texture {
        None => Vector3::new(0.0, 0.0, 0.0),
        Some(texture) => texture.getColorBilinear(u,v), // Do modification here
    };
    texture_color.norm()
}

pub fn bump_fragment_shader(payload: &FragmentShaderPayload) -> V3f {
    let ka = Vector3::new(0.005, 0.005, 0.005);
    let kd = payload.color;
    let ks = Vector3::new(0.7937, 0.7937, 0.7937);

    let l1 = Light {
        position: Vector3::new(20.0, 20.0, 20.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let l2 = Light {
        position: Vector3::new(-20.0, 20.0, 0.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let lights = vec![l1, l2];
    let amb_light_intensity = Vector3::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::new(0.0, 0.0, 10.0);

    let p = 150.0;

    let mut normal = payload.normal;
    let point = payload.view_pos;
    let color = payload.color;

    let (kh, kn) = (0.2, 0.1);

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)

    let t = Vector3::new(normal.x * normal.y / (normal.x * normal.x + normal.z * normal.z).sqrt(), (normal.x * normal.x + normal.z * normal.z).sqrt(), normal.z * normal.y / (normal.x * normal.x + normal.z * normal.z).sqrt());
    let b = normal.cross(&t);
    let TBN = Matrix3::from_columns(&[t,b,normal]);

    let u = payload.tex_coords.x;
    let v = payload.tex_coords.y;
    let dU = kh * kn * (h(payload, u + 1.0 / payload.texture.clone().unwrap().width as f64, v) - h(payload, u, v));
    let dV = kh * kn *(h(payload, u, v + 1.0 / payload.texture.clone().unwrap().height as f64) - h(payload, u, v));
    let ln = Vector3::new(-dU, -dV, 1.0);
    normal = (TBN * ln).normalize();

    let mut result_color = Vector3::zeros();
    result_color = normal;

    result_color * 255.0
}

pub fn displacement_fragment_shader(payload: &FragmentShaderPayload) -> V3f {
    let ka = Vector3::new(0.005, 0.005, 0.005);
    let kd = payload.color;
    let ks = Vector3::new(0.7937, 0.7937, 0.7937);

    let l1 = Light {
        position: Vector3::new(20.0, 20.0, 20.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let l2 = Light {
        position: Vector3::new(-20.0, 20.0, 0.0),
        intensity: Vector3::new(500.0, 500.0, 500.0),
    };
    let lights = vec![l1, l2];
    let amb_light_intensity = Vector3::new(10.0, 10.0, 10.0);
    let eye_pos = Vector3::new(0.0, 0.0, 10.0);

    let p = 150.0;

    let mut normal = payload.normal.normalize();
    let mut point = payload.view_pos;
    let color = payload.color;

    let (kh, kn) = (0.2, 0.1);

    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

    let t = Vector3::new(normal.x * normal.y / (normal.x * normal.x + normal.z * normal.z).sqrt(), (normal.x * normal.x + normal.z * normal.z).sqrt(), normal.z * normal.y / (normal.x * normal.x + normal.z * normal.z).sqrt());
    let b = normal.cross(&t);
    let TBN = Matrix3::from_columns(&[t,b,normal]);

    let u = payload.tex_coords.x;
    let v = payload.tex_coords.y;
    let dU = kh * kn * (h(payload, u + 1.0 / payload.texture.clone().unwrap().width as f64, v) - h(payload, u, v));
    let dV = kh * kn *(h(payload, u, v + 1.0 / payload.texture.clone().unwrap().height as f64) - h(payload, u, v));
    let ln = Vector3::new(-dU, -dV, 1.0);
    point = point + normal * kn * h(payload, u , v);
    normal = (TBN * ln).normalize();

    let mut result_color = Vector3::zeros();
    for light in lights {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.
        let l = (light.position - point).normalize();
        let v = (eye_pos - point).normalize();
        let h = (l + v).normalize();
        let n = normal.normalize();

        let cos_theta = l.dot(&n);
        let diffusely_reflected_light = elem_mul(kd, light.intensity / length_squared(light.position - point)) * cos_theta.max(0.0);
        result_color += diffusely_reflected_light;
        let cos_alpha = h.dot(&n);
        let specularly_reflected_light = elem_mul(ks, light.intensity / length_squared(light.position - point)) * cos_alpha.max(0.0).powf(p);
        result_color += specularly_reflected_light;
        let reflected_ambient_light = elem_mul(ka,amb_light_intensity);
        result_color += reflected_ambient_light;
    }

    result_color * 255.0
}