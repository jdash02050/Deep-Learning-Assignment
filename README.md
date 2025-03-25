Ensure Python is up to date:
python --version

Create a virtual environment:
  Windows:
  python -m venv venv
  .\venv\Scripts\activate

  Linux/macOS:
  python3 -m venv venv
  source venv/bin/activate

Install Dependencies:
pip install -r requirements.txt

Prepare the Dataset:
Download the dataset; The structure should look like this:

eel4810-dataset/eel4810-dataset/

    sub01/
        [CSV files here]
    sub02/
        [CSV files here]
    sub03/
        [CSV files here]
    sub05/
        [CSV files here]

Run the training script:

python src/train.py --base_folder "path/to/eel4810-dataset/eel4810-dataset" --normalize True --batch_size 32 --learning_rate 0.01 --optimizer adam --weight_init random --l2_reg 0.0

Pass these arguments with the function to change various aspects about the model's setup\

--base_folder: The path to the folder containing the dataset (eel4810-dataset).
--normalize: Set this to True to normalize the features.
--batch_size: The batch size for training (default is 32).
--learning_rate: The learning rate for the optimizer (default is 0.01).
--optimizer: The optimizer to use, such as adam, sgd, etc.
--weight_init: The weight initialization method (can be random or xavier).
--l2_reg: The L2 regularization coefficient (default is 0.0).

For example, to train with the Adam optimizer, Xavier weight initialization, and L2 regularization:
python src/train.py --base_folder "path/to/eel4810-dataset/eel4810-dataset" --normalize True --batch_size 32 --learning_rate 0.01 --optimizer adam --weight_init xavier --l2_reg 0.01



Testing the Model:
After the model finishes training, it will be saved as a .pth file
Test the model using the following:
python src/test.py --model_path "path/to/saved_model.pth" --test_data "path/to/test_data.csv"

where
--model_path: Path to the trained model file.
--test_data: Path to the test data (e.g., a CSV file).

This takes a VERY long time running on a CPU (which is all that I had available)
