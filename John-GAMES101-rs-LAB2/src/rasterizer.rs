use std::collections::HashMap;

use nalgebra::{Matrix4, Vector3, Vector4};
use crate::triangle::Triangle;

#[allow(dead_code)]
pub enum Buffer {
    Color,
    Depth,
    Both,
}

#[allow(dead_code)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Default, Clone)]
pub struct Rasterizer {
    model: Matrix4<f64>,
    view: Matrix4<f64>,
    projection: Matrix4<f64>,
    pos_buf: HashMap<usize, Vec<Vector3<f64>>>,
    ind_buf: HashMap<usize, Vec<Vector3<usize>>>,
    col_buf: HashMap<usize, Vec<Vector3<f64>>>,

    frame_buf: Vec<Vector3<f64>>,
    depth_buf: Vec<f64>,
    /*  You may need to uncomment here to implement the MSAA method  */
    // frame_sample: Vec<Vector3<f64>>,
    // depth_sample: Vec<f64>,
    width: u64,
    height: u64,
    next_id: usize,
}

#[derive(Clone, Copy)]
pub struct PosBufId(usize);

#[derive(Clone, Copy)]
pub struct IndBufId(usize);

#[derive(Clone, Copy)]
pub struct ColBufId(usize);

impl Rasterizer {
    pub fn new(w: u64, h: u64) -> Self {
        let mut r = Rasterizer::default();
        r.width = w;
        r.height = h;
        r.frame_buf.resize((w * h) as usize, Vector3::zeros());
        r.depth_buf.resize((w * h) as usize, 0.0);
        r
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        ((self.height - 1 - y as u64) * self.width + x as u64) as usize
    }

    fn set_pixel(&mut self, point: &Vector3<f64>, color: &Vector3<f64>) {
        let ind = (self.height as f64 - 1.0 - point.y) * self.width as f64 + point.x;
        self.frame_buf[ind as usize] = *color;
    }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
            }
            Buffer::Depth => {
                self.depth_buf.fill(f64::MAX);
            }
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MAX);
            }
        }
    }

    pub fn set_model(&mut self, model: Matrix4<f64>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f64>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f64>) {
        self.projection = projection;
    }

    fn get_next_id(&mut self) -> usize {
        let res = self.next_id;
        self.next_id += 1;
        res
    }

    pub fn load_position(&mut self, positions: &Vec<Vector3<f64>>) -> PosBufId {
        let id = self.get_next_id();
        self.pos_buf.insert(id, positions.clone());
        PosBufId(id)
    }

    pub fn load_indices(&mut self, indices: &Vec<Vector3<usize>>) -> IndBufId {
        let id = self.get_next_id();
        self.ind_buf.insert(id, indices.clone());
        IndBufId(id)
    }

    pub fn load_colors(&mut self, colors: &Vec<Vector3<f64>>) -> ColBufId {
        let id = self.get_next_id();
        self.col_buf.insert(id, colors.clone());
        ColBufId(id)
    }

    pub fn draw(&mut self, pos_buffer: PosBufId, ind_buffer: IndBufId, col_buffer: ColBufId, _typ: Primitive) {
        let buf = &self.clone().pos_buf[&pos_buffer.0];
        let ind: &Vec<Vector3<usize>> = &self.clone().ind_buf[&ind_buffer.0];
        let col = &self.clone().col_buf[&col_buffer.0];

        let f1 = (50.0 - 0.1) / 2.0;
        let f2 = (50.0 + 0.1) / 2.0;

        let mvp = self.projection * self.view * self.model;

        for i in ind {
            let mut t = Triangle::new();
            let mut v =
                vec![mvp * to_vec4(buf[i[0]], Some(1.0)), // homogeneous coordinates
                     mvp * to_vec4(buf[i[1]], Some(1.0)), 
                     mvp * to_vec4(buf[i[2]], Some(1.0))];
    
            for vec in v.iter_mut() {
                *vec = *vec / vec.w;
            }
            for vert in v.iter_mut() {
                vert.x = 0.5 * self.width as f64 * (vert.x + 1.0);
                vert.y = 0.5 * self.height as f64 * (vert.y + 1.0);
                vert.z = vert.z * f1 + f2;
            }
            for j in 0..3 {
                // t.set_vertex(j, Vector3::new(v[j].x, v[j].y, v[j].z));
                t.set_vertex(j, v[j].xyz());
                t.set_vertex(j, v[j].xyz());
                t.set_vertex(j, v[j].xyz());
            }
            let col_x = col[i[0]];
            let col_y = col[i[1]];
            let col_z = col[i[2]];
            t.set_color(0, col_x[0], col_x[1], col_x[2]);
            t.set_color(1, col_y[0], col_y[1], col_y[2]);
            t.set_color(2, col_z[0], col_z[1], col_z[2]);

            self.rasterize_triangle(&t);
        }
    }

    //无抗锯齿
    /*pub fn rasterize_triangle(&mut self, t: &Triangle) {
        for x in 0..=self.width{
            for y in 0..=self.height{
                if inside_triangle(x as f64 + 0.5 , y as f64 + 0.5 , &t.v) {
                    let x1 = x as f64 + 0.5 - t.v[0].x;
                    let y1 = y as f64 + 0.5 - t.v[0].y;
                    let x2 = t.v[2].x - t.v[0].x;
                    let y2 = t.v[2].y - t.v[0].y;
                    let x3 = t.v[1].x - t.v[0].x;
                    let y3 = t.v[1].y - t.v[0].y;
                    let u = (x1*y3-x3*y1)/(x2*y3-x3*y2);
                    let v = (x1*y2-x2*y1)/(x3*y2-x2*y3);
                    let depth =-(t.v[0].z + u * (t.v[2].z - t.v[0].z) + v * (t.v[1].z - t.v[0].z));
                    if self.depth_buf[self.get_index(x as usize, y as usize)] > depth {
                        self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &t.get_color());
                        let position = self.get_index(x as usize, y as usize);
                        self.depth_buf[position] = depth;
                    }
                }
            }
        }
    }*/

    //MSAA
    pub fn rasterize_triangle(&mut self, t: &Triangle) {
        for x in 0..=self.width{
            for y in 0..=self.height{
                let mut num = 0.0;
                if inside_triangle(x as f64 + 0.25 , y as f64 +0.25 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.25 , y as f64 +0.75 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.75 , y as f64 +0.25 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.75 , y as f64 +0.75 , &t.v) {
                    num+=1.0;
                }
                if num > 0.0 {
                    let x1 = x as f64 + 0.5 - t.v[0].x;
                    let y1 = y as f64 + 0.5 - t.v[0].y;
                    let x2 = t.v[2].x - t.v[0].x;
                    let y2 = t.v[2].y - t.v[0].y;
                    let x3 = t.v[1].x - t.v[0].x;
                    let y3 = t.v[1].y - t.v[0].y;
                    let u = (x1*y3-x3*y1)/(x2*y3-x3*y2);
                    let v = (x1*y2-x2*y1)/(x3*y2-x2*y3);
                    let depth =-(t.v[0].z + u * (t.v[2].z - t.v[0].z) + v * (t.v[1].z - t.v[0].z));
                    if self.depth_buf[self.get_index(x as usize, y as usize)] > depth {
                        self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &Vector3::new(t.get_color().x * num / 4.0, t.get_color().y * num / 4.0, t.get_color().z * num / 4.0));
                        let position = self.get_index(x as usize, y as usize);
                        self.depth_buf[position] = depth;
                    }
                }
            }
        }
    }

    //TAA
    /*pub fn rasterize_triangle(&mut self, t: &Triangle) {
        for x in 0..=self.width{
            for y in 0..=self.height{
                let mut num = 0.0;
                if inside_triangle(x as f64 + 0.25 , y as f64 +0.25 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.25 , y as f64 +0.75 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.75 , y as f64 +0.25 , &t.v) {
                    num+=1.0;
                }
                if inside_triangle(x as f64 + 0.75 , y as f64 +0.75 , &t.v) {
                    num+=1.0;
                }
                if num > 0.0 {
                    let x1 = x as f64 + 0.5 - t.v[0].x;
                    let y1 = y as f64 + 0.5 - t.v[0].y;
                    let x2 = t.v[2].x - t.v[0].x;
                    let y2 = t.v[2].y - t.v[0].y;
                    let x3 = t.v[1].x - t.v[0].x;
                    let y3 = t.v[1].y - t.v[0].y;
                    let u = (x1*y3-x3*y1)/(x2*y3-x3*y2);
                    let v = (x1*y2-x2*y1)/(x3*y2-x2*y3);
                    let depth =-(t.v[0].z + u * (t.v[2].z - t.v[0].z) + v * (t.v[1].z - t.v[0].z));
                    if self.depth_buf[self.get_index(x as usize, y as usize)] > depth {
                        self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &Vector3::new(t.get_color().x * num / 4.0, t.get_color().y * num / 4.0, t.get_color().z * num / 4.0));
                        let position = self.get_index(x as usize, y as usize);
                        self.depth_buf[position] = depth;
                    }
                }
            }
        }
    }*/




    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }
}

fn to_vec4(v3: Vector3<f64>, w: Option<f64>) -> Vector4<f64> {
    Vector4::new(v3.x, v3.y, v3.z, w.unwrap_or(1.0))
}

fn inside_triangle(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> bool {
    /*  implement your code here  */
    let x1 = x - v[0].x;
    let y1 = y - v[0].y;
    let x2 = v[2].x - v[0].x;
    let y2 = v[2].y - v[0].y;
    let x3 = v[1].x - v[0].x;
    let y3 = v[1].y - v[0].y;
    let u = (x1*y3-x3*y1)/(x2*y3-x3*y2);
    let v = (x1*y2-x2*y1)/(x3*y2-x2*y3);

    if u > 0.0 && v > 0.0 && u + v < 1.0 {
        return true;
    }
    false
}

fn compute_barycentric2d(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> (f64, f64, f64) {
    let c1 = (x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * y + v[1].x * v[2].y - v[2].x * v[1].y)
        / (v[0].x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * v[0].y + v[1].x * v[2].y - v[2].x * v[1].y);
    let c2 = (x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * y + v[2].x * v[0].y - v[0].x * v[2].y)
        / (v[1].x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * v[1].y + v[2].x * v[0].y - v[0].x * v[2].y);
    let c3 = (x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * y + v[0].x * v[1].y - v[1].x * v[0].y)
        / (v[2].x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * v[2].y + v[0].x * v[1].y - v[1].x * v[0].y);
    (c1, c2, c3)
}