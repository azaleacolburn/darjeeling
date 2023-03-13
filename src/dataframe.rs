use std::fmt::Debug;

#[derive(Debug)]
pub struct DataFrame<'a, T: Copy + Debug> {
    frame: Vec<Vec<Point<T>>>,
    row_labels: Vec<&'a str>,
    col_labels: Vec<&'a str>
}

#[derive(Debug, Clone)]
pub struct Point<T: Copy + Debug + Clone> {
    row: usize,
    col: usize,
    val: T
}

impl<'a, T: Copy + Debug> DataFrame<'a, T> {

    pub fn new(frame: Vec<Vec<T>>, row_labels: Vec<&'a str>, col_labels: Vec<&'a str>) -> DataFrame<'a, T> {

        let mut proper_frame: Vec<Vec<Point<T>>> = vec![];
        for i in 0..frame.len() {

            proper_frame.push(Point::point_vector(i, frame[i].clone()));
        }

        DataFrame { frame: proper_frame, row_labels, col_labels }
    }

    pub fn value_at_index(&self, x: usize, y: usize) -> &T {

        &&self.frame[x][y].val
    }

    pub fn index_at_labels(&self, row: &str, col: &str) -> Result<(usize, usize), &str>{

        let row_index = match DataFrame::<'a, T>::label_index(row, &self.row_labels) {

            Ok(i) => i,
            Err(error) => return Err(error)
        };
        let col_index = match DataFrame::<'a, T>::label_index(col, &self.col_labels) {

            Ok(i) => i,
            Err(error) => return Err(error)

        };

        Ok((row_index, col_index))
    }

    pub fn value_at_labels(&self, row: &str, col: &str) ->  Result<&T, &str> {

        let row_index: usize = match DataFrame::<'a, T>::label_index(row, &self.row_labels) {

            Ok(i) => i,
            Err(error) => return Err(error)
        };
        let col_index: usize = match DataFrame::<'a, T>::label_index(col, &self.col_labels) {

            Ok(i) => i,
            Err(error) => return Err(error)

        };

        Ok(self.value_at_index(row_index, col_index))
    }

    pub fn add_row(&mut self, label: &'a str, content: Vec<Point<T>>) {

        // Checks to make sure the dataframe is still valid
        assert_eq!(self.row_labels.len(), self.frame.len());

        self.row_labels.push(label);
        self.frame.push(content);

        assert_eq!(self.col_labels.len(), self.col_labels.len());
    }

    pub fn add_col(&mut self, label: &'a str, content: Vec<Point<T>>) -> Result<(), String> {
        
        // Checks to make sure the dataframe is still valid
        assert_eq!(self.col_labels.len(), self.col_labels.len());
        
        let index:usize = match self.col_labels.iter().position(|x| *x == label) {

            Some(i) => i,
            None => {
                return Err(format!("This label doesn't exist: {:?}", label));
            }
        };
        self.col_labels.insert(index, label);
        self.frame.insert(self.frame.len() - 1, content);

        assert_eq!(self.col_labels.len(), self.col_labels.len());

        Ok(())
    }
    
    pub fn delete_row(&mut self, label: &'a str) -> Result<(), &str> {
        
        let index: usize = match self.row_labels.iter().position(|x| *x == label) {

            Some(i) => i,
            None => return Err("Row not found")
        };
        self.row_labels.remove(index);
        self.frame.remove(index);  
        
        Ok(())
    }

    pub fn delete_column<'b>(&'b mut self, label: &'a str) -> Result<(), &'b str> {

        let index: usize = match self.col_labels.iter().position(|x| *x == label) {

            Some(i) => i,
            None => return Err("Column not defined")
        };
        self.col_labels.remove(index);
        self.frame.remove(index);  
        
        Ok(())
    }

    pub fn label_index(label: &str, labels: &Vec<&str>) -> Result<usize, &'a str> {

        let mut col_num: Option<usize> = None;

        for i in 0..labels.len() {
            if labels[i] == label { 
                col_num = Some(i);
                break;
            }
        }

        match col_num {

            Some(n) => Ok(n),
            None => Err("This label doesn't exist")
        }
    }

    pub fn get_cols_len(&self) -> usize {

        self.col_labels.len()
    }

    pub fn get_rows_len(&self) -> usize {

        self.row_labels.len()
    }

    pub fn display(&self) {
        for _i in 0..self.row_labels[0].len() + 2 {
            print!(" ");
        }
        for i in &self.col_labels {
            print!("{:?} ", i)
        }
        print!("\n");
        for i in 0..self.row_labels.len() {
            print!("{:?} ", self.row_labels[i]);
            let longest = DataFrame::<'a, T>::find_longest(&self.row_labels);
            let shortest = DataFrame::<'a, T>::find_shortest(&self.row_labels);
            if longest != self.row_labels[i].len() {
                for _l in 0..longest - self.row_labels[0].len() {
                    print!(" ");
                }
            }
            for j in 0..self.col_labels.len() {
                print!("{:?}", self.frame[i][j].val);
                if longest != self.row_labels[i].len() && shortest != longest{
                    for _k in 0..self.col_labels[j].len() + 2 {
                        print!(" ");
                    }
                } else {
                    for _k in 0..self.col_labels[j].len() {
                        print!(" ");
                    }
                }
            }
            print!("\n");
        }
    }

    pub fn find_longest(vec: &Vec<&str>) -> usize {

        let mut longest_len: usize = vec[0].len();
        for i in 1..vec.len() {
            let length = vec[i].len();
            if length > longest_len { longest_len = length }
        }

        longest_len
    }

    pub fn find_shortest(vec: &Vec<&str>) -> usize {

        let mut shortest_len: usize = vec[0].len();
        for i in 1..vec.len() {
            let length = vec[i].len();
            if length < shortest_len { shortest_len = length; }
        }

        shortest_len
    }
     
}

impl<T: Copy + Debug> Point<T> {

    pub fn new(row: usize, col: usize, content: T) -> Point<T> {

        Point { row, col, val: content }
    }

    pub fn point_vector(col: usize, vector: Vec<T>) -> Vec<Point<T>> {

        let mut new_vector: Vec<Point<T>> = vec![];

        for row in 0..vector.len() {
            new_vector.push(Point::new(row, col, vector[row]));
        }

        new_vector
    }
}