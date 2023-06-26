use std::fmt::Display;

use crate::types::Types;

#[allow(dead_code)]

#[derive(Debug, Clone)]
/// Represents the input for the neural network
pub struct Input {
    pub inputs: Vec<f32>,
    pub answer: Option<Types>
}

impl Input  {

    /// Creates new input
    /// # Params
    /// - Inputs: A list of 32-bit floating point numbers.
    /// - Answer: A string, representing the correct answer(or cateogry) of this input.
    /// 
    /// # Examples
    /// This example is for one input into an XOR gate
    /// ```
    /// use darjeeling::input::Input;
    /// use darjeeling::types::Types;
    /// let inputs: Vec<f32> = vec![0.0,1.0];
    /// let answer: Types = Types::String("1".to_string());
    /// let formated_input: Input = Input::new(inputs, answer);
    /// ```
    pub fn new(inputs: Vec<f32>, answer: Option<Types>) -> Input {

        Input { inputs, answer}
    }

    // TODO: Write format_as_input function
}

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = String::from("");
        for i in 0..self.inputs.len() - 1 {
            buffer += format!("{},", self.inputs[i]).as_str();
        }
        buffer += format!("{}", self.inputs[self.inputs.len() - 1]).as_str();
        write!(f, "{}", buffer)
    }
}