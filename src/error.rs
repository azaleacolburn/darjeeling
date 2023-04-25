use core::fmt;
use std::any::TypeId;

#[derive(Debug)]
pub enum DarjeelingError<'a> {

    ColumnDoesNotExist(&'a str), 
    RowDoesNotExist(&'a str),
    PointDoesNotExist((&'a str, &'a str)),
    ReadModelFailed(String, std::io::Error),
    ReadModelFunctionFailed(String, Box<DarjeelingError<'a>>),
    WriteModelFailed(String),
    ModelNameAlreadyExists(String),
    InvalidFormatType(TypeId),

    UnknownError(std::io::Error)
}

impl<'a> fmt::Display for DarjeelingError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DarjeelingError::ColumnDoesNotExist(column) => write!(f, 
                "{:?} isn't a vali column", 
                column
            ),
            DarjeelingError::RowDoesNotExist(row) => write!(f, 
                "{:?} isn't a valid row", 
                row
            ),
            DarjeelingError::PointDoesNotExist((row, column)) => write!(f, 
                "There is no point at row: {:?}, column: {:?}", 
                row, column
            ),
            DarjeelingError::ReadModelFailed(model_name, error) => write!(f,
                "Unable to read model {:?}, \n Hint: Double check the model_name \n Error Message: {:?}",
                model_name, error
            ),
            DarjeelingError::ReadModelFunctionFailed(model_name,error ) => write!(f,
                "The read model function failed on the model {:?} \n The error given by the ReadModelFunction was {:?}",
                model_name, error
            ),
            DarjeelingError::WriteModelFailed(model_name) => write!(f,
                "Unable to write model {:?}, \n Hint: This is probably because the random name already exists, try saving it again",
                model_name.as_str()
            ),
            DarjeelingError::ModelNameAlreadyExists(model_name) => write!(f,
                "Model name {:?} already exists, \n Hint: Try saving it again",
                model_name.as_str()
            ),
            DarjeelingError::InvalidFormatType(type_id) => write!(f,
                "We couldn't format this value because the type: {:?} wasn't valid",
                type_id
            ),
            DarjeelingError::UnknownError(error) => write!(f,
                "Non-Darjeeling error encountered: \n {:?}",
                error
            )
        }
    }
}