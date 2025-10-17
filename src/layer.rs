use ndarray::{arr2, Array1, Array2};
use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    pub cached_outputs: Array1<f32>,
    pub categories: Box<[String]>,
    pub correct_answers: Box<[f32]>,
}

impl Layer {
    pub fn new_neuron_layer(size: usize, previous_layer_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let generate_links = |_: usize| {
            (0..previous_layer_size)
                .map(|_| rng.gen_range(-0.05..0.05))
                .collect::<Vec<f32>>()
        };

        Self::generate_layer(generate_links, size, previous_layer_size)
    }
    pub fn new_input_layer(size: usize) -> Self {
        let generate_links = |_: usize| vec![0.];

        Self::generate_layer(generate_links, size, 1)
    }

    fn generate_layer(
        generate_links: impl FnMut(usize) -> Vec<f32>,
        size: usize,
        previous_layer_size: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let weights_vec = (0..size).flat_map(generate_links).collect::<Vec<f32>>();
        let weights = Array2::from_shape_vec((size, previous_layer_size), weights_vec)
            .expect("Invalid dimensions for weights matrix");

        let biases = Array1::from_vec((0..size).map(|_| rng.gen_range(-0.05..0.05)).collect());
        let cached_outputs: Array1<f32> = Array1::from_vec(vec![0.; size]);
        let categories = vec![String::new(); size].into_boxed_slice();
        let correct_answers = vec![0.; size].into_boxed_slice(); // this will get changed each epoch

        Self {
            weights,
            biases,
            cached_outputs,
            correct_answers,
            categories,
        }
    }

    pub fn feedforward_layer(
        &mut self,
        previous_layer: &Layer,
        activation_function: fn(f32) -> f32,
    ) {
        self.cached_outputs = (self.cached_outputs.dot(&previous_layer.weights) + &self.biases)
            .map(|n| activation_function(*n));
    }
}

impl Serialize for Layer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        todo!()
    }
}
impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}
