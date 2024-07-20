use crate::activation::ActivationFunction;
use crate::error::DarjeelingError;
use crate::series::Series;

pub trait NeuralNetwork {
    fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        answer_nodes: usize,
        hidden_layers: usize,
        activation_function: Option<ActivationFunction>,
    ) -> Self;

    fn train(
        &mut self,
        data: &Box<[Series]>,
        categories: Box<[String]>,
        learning_rate: f32,
        name: &str,
        target_err_percent: f32,
        write: bool,
    ) -> Result<(Option<String>, f32, f32), DarjeelingError>;

    fn test(
        &mut self,
        data: &Box<[Series]>,
        categories: Box<[String]>,
    ) -> Result<Vec<String>, DarjeelingError>;
}
