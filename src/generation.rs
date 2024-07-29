use crate::{
    activation::ActivationFunction, categorize::CatNetwork, dbg_println, error::DarjeelingError,
    neural_network::NeuralNetwork, node::Node, series::Series, utils::RandomIter, DEBUG,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

/// The generation Neural Network struct
#[derive(Debug, Serialize, Deserialize)]
pub struct GenNetwork {
    node_array: Box<[Box<[Node]>]>,
    activation_function: Option<ActivationFunction>,
}
#[warn(clippy::unwrap_in_result)]
impl GenNetwork {
    /// Constructor function for the neural network
    /// Fills a Neural Network's node_array with empty nodes.
    /// Initializes random starting link and bias weights between -.5 and .5
    ///
    /// ## Params
    /// - Inputs: The number of sensors in the input layer
    /// - Hidden: The number of hidden nodes in each layer
    /// - Answer: The number of answer nodes, or possible categories
    /// - Hidden Layers: The number of different hidden layers
    ///
    /// ## Examples
    /// ``` rust
    /// use darjeeling::{
    ///     activation::ActivationFunction,
    ///     generation::GenNetwork
    /// };
    ///
    /// let inputs: i32 = 10;
    /// let hidden: i32 = 40;
    /// let answer: i32 = 2;
    /// let hidden_layers: i32 = 1;
    /// let mut net = GenNetwork::new(inputs, hidden, answer, hidden_layers, ActivationFunction::Sigmoid);
    /// ```
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        answer_nodes: usize,
        hidden_layers: usize,
        activation_function: Option<ActivationFunction>,
    ) -> GenNetwork {
        let mut rng = rand::thread_rng();

        let mut node_array: Vec<Box<[Node]>> = vec![];

        let input_row: Box<[Node]> = (0..input_nodes)
            .map(|_| Node::new(vec![1.00; hidden_nodes].into_boxed_slice(), 1.0))
            .collect::<Box<[Node]>>();
        node_array.push(input_row);

        for _ in 0..hidden_layers + 1 {
            let hidden_vec: Box<[Node]> = (0..hidden_nodes)
                .into_iter()
                .map(|_| {
                    Node::new(
                        vec![rng.gen_range(-0.5..0.5); answer_nodes].into_boxed_slice(),
                        rng.gen_range(-0.5..0.5),
                    )
                })
                .collect::<Box<[Node]>>();
            node_array.push(hidden_vec);
        }

        // TODO: Make these link weights None or smth
        let answer_row: Box<[Node]> = (0..answer_nodes)
            .into_iter()
            .map(|_| Node::new(vec![0.00; 1].into_boxed_slice(), rng.gen_range(-0.5..0.5)))
            .collect::<Box<[Node]>>();

        node_array.push(answer_row);

        GenNetwork {
            node_array: node_array.into_boxed_slice(),
            activation_function,
        }
    }

    /// Trains a neural model to generate new data formatted as inputs, based on the given data
    ///
    /// ## Params
    /// - Data: List of inputs to be trained on
    /// - Learning Rate: The modifier that is applied to link weights as they're adjusted.
    /// Try fiddling with this one, but -1.5 - 1.5 is recommended to start.
    /// - Name: The model name
    /// - Max Cycles: The maximum number of epochs the training will run for.
    /// - Distinguishing Learning Rate: The learning rate for the distinguishing model.
    /// - Distinguishing Hidden Neurons: The number of hidden neurons in each layer of the distinguishing model.
    /// - Distinguising Hidden Layers: The number of hidden layers in the distinguishing model.
    /// - Distinguishing Activation: The activation function of the distinguishing model.
    /// - Distinguishing Target Error Percent: The error percentage at which the distinguishing models will stop training.
    ///
    /// ## Returns
    /// The fallible name of the model that this neural network trained
    ///
    /// ## Err
    /// ### WriteModelFailed
    /// There was a problem when saving the model to a file
    ///
    /// ### ModelNameAlreadyExists
    /// The random model name chosen already exists
    /// Change the name or retrain
    ///
    /// ### RemoveModelFailed
    /// Every time a new distinguishing model is written to the project folder, the previous one has to be removed.
    /// This removal failed,
    ///
    /// ### DistinguishingModel
    /// The distinguishing model training failed.
    ///
    /// ### UnknownError
    /// Not sure what happened, but something failed
    ///
    /// Make an issue on the [darjeeling](https://github.com/Ewie21/darjeeling) GitHub page
    /// Or contact me at elocolburn@comcast.net
    ///
    /// ## TODO: Refactor to pass around the neural net, not the model name
    ///
    /// ## Examples
    /// ```ignore
    /// use darjeeling::{
    ///     generation::GenNetwork,
    ///     activation::ActivationFunction,
    ///     input::Input,
    ///     // This file may not be avaliable
    ///     // Everything found here will be hyper-specific to your project.
    ///     tests::{categories_str_format, file}
    /// };
    ///
    /// // A file with data
    /// // To make sure the networked is properly trained, make sure it follows some sort of pattern
    /// // This is just sample data, for accurate results, around 3800 datapoints is needed
    /// // 1 2 3 4 5 6 7 8
    /// // 3 2 5 4 7 6 1 8
    /// // 0 2 5 4 3 6 1 8
    /// // 7 2 3 4 9 6 1 8
    /// // You also need to write the file input function
    /// // Automatic file reading and formatting function coming soon
    /// let mut data: Vec<Input> = file();
    /// let mut net = GenNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
    /// let model_name: String = net.learn(&mut data, 0.5, "gen", 100, 0.5, 10, 1, ActivationFunction::Sigmoid, 99.0).unwrap();
    /// let new_data: Vec<Input> = net.test(data).unwrap();
    /// ```
    pub fn learn(
        // Frankly this whole function is disgusting and needs to be burned; I concur from the future
        &mut self,
        data: &Box<[Box<[f32]>]>,
        learning_rate: f32,
        name: &str,
        max_cycles: usize,
        distinguising_learning_rate: f32,
        distinguising_hidden_neurons: usize,
        distinguising_hidden_layers: usize,
        distinguising_activation: ActivationFunction,
        distinguishing_target_err_percent: f32,
    ) -> Result<String, DarjeelingError> {
        let mut epochs: f32 = 0.0;
        let mut distinguishing_model = CatNetwork::new(
            self.node_array.last().expect("Network has no layers").len(),
            distinguising_hidden_neurons,
            2,
            distinguising_hidden_layers,
            Some(distinguising_activation),
        );
        let activation_function = self.activation_function.unwrap();

        let mut outputs: Vec<Box<[f32]>> = vec![];
        for _ in 0..max_cycles {
            let data_iter = RandomIter::new(data);

            // Train generation network
            for line in data_iter {
                dbg_println!("Training Checkpoint One Passed");
                self.push_downstream(line, activation_function);

                let answer_layer = self.node_array.last().expect("Network has no layers");
                outputs.push(
                    answer_layer
                        .iter()
                        .map(|node| {
                            node.cached_output
                                .expect("Answer layer node has no cached output")
                        })
                        .collect::<Box<[f32]>>(),
                );
            }

            let series_data: Box<[Series]> = data
                .into_iter()
                .map(|line| Series::new(line.clone(), ""))
                .collect();

            let mse: f32 = match distinguishing_model.train(
                &series_data,
                vec!["real".to_string(), "generated".to_string()].into_boxed_slice(),
                distinguising_learning_rate,
                &("distinguishing".to_owned() + &name),
                distinguishing_target_err_percent,
                false,
            ) {
                Ok((_name, _err_percent, errmse)) => errmse,
                Err(error) => return Err(error),
            };

            self.backpropogate(learning_rate, activation_function);
            epochs += 1.0;
            println!("Epoch: {:?}", epochs);
        }

        Ok(self.write_model(&name)?)
    }

    pub fn test(&mut self, data: &Box<[Box<[f32]>]>) -> Result<Box<[Box<[f32]>]>, DarjeelingError> {
        let data_iter = RandomIter::new(data);
        let activation_function = self.activation_function.unwrap();

        Ok(data_iter.map(|line|{
            self.push_downstream(line, activation_function);
            self.node_array.last().expect("Network has no layers")
                .iter()
                .map(|node| {
                    node.cached_output
                        .expect("Answer layer node does not have cached output")
                })
                .collect()
        }).collect())
    }

    /// Passes in data to the sensors, pushs data 'downstream' through the network
    fn push_downstream(&mut self, data: &Box<[f32]>, activation_function: ActivationFunction) {
        // Pass data to the input layer
        if let Some(input_layer) = self.node_array.first_mut() {
            input_layer.iter_mut().enumerate().for_each(|(i, node)| {
                node.cached_output = Some(data[i]);
            });
        }

        // Push forward hidden and output layers
        for layer_i in 1..self.node_array.len() {
            // Clone the cached outputs from the previous layer
            let prev_cached_outputs: Box<[f32]> = self.node_array[layer_i - 1]
                .iter()
                .map(|node| {
                    node.cached_output
                        .expect("Previous nodes do not have cached outputs")
                })
                .collect();

            let layer = self.node_array[layer_i].iter_mut();

            layer.for_each(|node| {
                prev_cached_outputs
                    .iter()
                    .enumerate()
                    .for_each(|(prev_node_i, prev_output)| {
                        node.link_vals[prev_node_i] = *prev_output;
                    });

                node.output(activation_function);
            });
        }
    }
    /// Analyses the chosen answer node's result.
    /// Also increments sum and count
    fn self_analysis(
        &self,
        epochs: &mut Option<f32>,
        sum: &mut f32,
        count: &mut f32,
        series: &Series,
        mse: &mut f32,
    ) -> (String, Option<f32>) {
        dbg_println!("answer {}", self.node_array.len() - 1);
        dbg_println!("largest index {}", self.largest_node());
        dbg_println!("{:?}", self);

        let answer_layer = self.node_array.last().expect("Network has no layer");
        let brightest_node: &Node = &answer_layer[self.largest_node()];
        let brightness: f32 = brightest_node.cached_output.unwrap();

        dbg_println!(
            "Category: {:?} \nBrightness: {:?}",
            brightest_node.category.as_ref().unwrap(),
            brightness
        );

        if brightest_node.category.clone().unwrap() == series.answer {
            dbg_println!("Sum++");
            *sum += 1.0;
        }

        *count += 1.0;

        match epochs {
            Some(epochs) => {
                // This won't happen during testing
                if *epochs % 10.0 != 0.0 || *epochs == 0.0 {
                    return (brightest_node.category.clone().unwrap(), None);
                }
                println!("\n-------------------------\n");
                println!("Epoch: {:?}", epochs);
                println!(
                    "Category: {:?} \nBrightness: {:?}",
                    brightest_node.category.as_ref().unwrap(),
                    brightness
                );
                if DEBUG {
                    let dimest_node: &Node =
                        &answer_layer[answer_layer.len() - self.largest_node() - 1];
                    println!(
                        "Chosen category: {:?} \nDimest Brightness: {:?}",
                        dimest_node.category.as_ref().unwrap(),
                        dimest_node.cached_output.unwrap()
                    );
                }
                (brightest_node.category.clone().unwrap(), None)
            }
            None => (
                brightest_node.category.clone().unwrap(),
                Some(CatNetwork::calculate_err_for_generation_model(
                    mse,
                    brightest_node,
                )),
            ),
        }
    }

    fn calculate_err_for_generation_model(mse: &mut f32, node: &Node) -> f32 {
        *mse += f32::powi(
            node.correct_answer.unwrap() - node.cached_output.unwrap(),
            2,
        );
        *mse
    }

    /// Finds the index and the brightest node in an array and returns it
    fn largest_node(&self) -> usize {
        let mut largest_index = 0;
        let answer_layer = self.node_array.last().expect("Network has no layers");
        for (i, node) in answer_layer.iter().enumerate() {
            if node.cached_output > answer_layer[largest_index].cached_output {
                largest_index = i;
            }
        }

        largest_index
    }
    /// Goes back through the network adjusting the weights of the all the neurons based on their error signal
    fn backpropogate(&mut self, learning_rate: f32, activation_function: ActivationFunction) {
        let hidden_layers = self.node_array.len() - 2;

        self.node_array
            .last_mut()
            .expect("Network has no layers")
            .iter_mut()
            .for_each(|node| {
                node.compute_answer_err_sig(activation_function);

                dbg_println!("Answer Node(Post Error Calc): {:?}", node);
            });

        self.adjust_hidden_weights(learning_rate, hidden_layers);

        self.node_array
            .last_mut()
            .expect("Network has no layers")
            .iter_mut()
            .for_each(|node| node.adjust_weights(learning_rate));
    }

    /// Adjusts the weights of all the hidden neurons in a network
    fn adjust_hidden_weights(&mut self, learning_rate: f32, hidden_layers: usize) {
        for layer in 1..=hidden_layers {
            for node in 0..self.node_array[layer].len() {
                self.node_array[layer][node].err_sig = Some(0.0);
                // link weights, err sigs
                let mut next_layer: Box<[(f32, f32)]> = self.node_array[layer + 1]
                    .iter()
                    .map(|next_node| {
                        (
                            next_node.link_weights[node],
                            next_node.err_sig.unwrap_or(0.00),
                        )
                    })
                    .collect::<Box<[(f32, f32)]>>();

                next_layer.iter_mut().for_each(|(next_weight, err_sig)| {
                    // This changes based on the activation function
                    self.node_array[layer][node].err_sig =
                        Some(*err_sig + (*err_sig * *next_weight));

                    dbg_println!("next err sig {:?}", err_sig);
                    dbg_println!("next weight {:?}", next_weight);
                });

                let hidden_result = self.node_array[layer][node].cached_output.unwrap();
                let multiplied_value = self.node_array[layer][node].err_sig.unwrap()
                    * (hidden_result)
                    * (1.0 - hidden_result);

                dbg_println!("New hidden errsig multiply: {:?}", multiplied_value);

                self.node_array[layer][node].err_sig = Some(multiplied_value);

                dbg_println!("\nLayer: {:?}", layer);
                dbg_println!("Node: {:?}", node);

                self.node_array[layer][node].adjust_weights(learning_rate);
            }
        }
    }

    /// Serializes a trained model so it can be used later
    ///
    /// ## Returns
    /// The name of the model
    ///
    /// ## Err
    /// ### WriteModelFailed:
    /// Wraps the models name
    /// ### UnknownError:
    /// Wraps error
    ///  
    pub fn write_model(&mut self, name: &str) -> Result<String, DarjeelingError> {
        let bin = match bincode::serialize(self) {
            Ok(v) => v,
            Err(err) => return Err(DarjeelingError::SaveModelFailed(err.to_string())),
        };

        let mut rng = rand::thread_rng();
        let num = rng.gen_range(0..i32::MAX);

        let model_name = format!("{}_{}.darj", name, num);
        match fs::write(name, bin) {
            Ok(_) => Ok(model_name),
            Err(err) => Err(DarjeelingError::SaveModelFailed(err.to_string())),
        }
    }

    /// Reads a file containing a serizalized Categorization Network
    ///
    /// ## Params
    /// - Model Name: The name(or more helpfully the path) of the model to be read
    ///
    /// ## Returns
    /// A neural network read from a serialized .darj file
    ///
    /// ## Err
    /// If the file cannnot be read, or if the file does not contain a valid serialized Neural Network
    pub fn read_model(model_name: String) -> Result<GenNetwork, DarjeelingError> {
        let read_bin = match fs::read(&model_name) {
            Ok(bin) => bin,
            Err(err) => return Err(DarjeelingError::LoadModelFailed(err.to_string())),
        };

        match bincode::deserialize(&read_bin) {
            Ok(net) => Ok(net),
            Err(err) => Err(DarjeelingError::LoadModelFailed(err.to_string())),
        }
    }

    pub fn set_activation_func(&mut self, new_activation_function: ActivationFunction) {
        self.activation_function = Some(new_activation_function);
    }
}
