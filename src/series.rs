use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug, Clone)]

/// Represents the training inputs to a neural network
pub struct Series {
    pub data: Box<[f32]>,
    pub answer: String,
}

impl Series {
    /// Creates new input
    /// # Params
    /// - Inputs: A list of 32-bit floating point numbers.
    /// - Answer: A string, representing the correct answer(or cateogry) of this input.
    ///
    /// # Examples
    /// This example is for one input into an XOR gate
    /// ```
    /// use darjeeling::series::Series;
    /// let inputs = vec![0.0,1.0];
    /// let answer = String::from("1");
    /// let formated_input = Series::new(inputs, answer);
    /// ```
    pub fn new<T, U>(data: T, answer: U) -> Series
    where
        T: Into<Box<[f32]>>,
        U: ToString,
    {
        Series {
            data: data.into(),
            answer: answer.to_string(),
        }
    }
}

impl Display for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = String::from("");
        for i in 0..self.data.len() - 1 {
            buffer += format!("{},", self.data[i]).as_str();
        }
        buffer += format!("{}", self.data.last().expect("Series has no elements")).as_str();
        write!(f, "{}", buffer)
    }
}
