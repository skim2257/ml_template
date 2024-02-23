from argparse import ArgumentParser

def parser():
    parser = ArgumentParser("imgtools Automatic Processing Pipeline.")

    # arguments
    parser.add_argument("--dataset_name", type=str, default="iris",
                        help="Which dataset do you want to train on?")
    
    parser.add_argument("--model_type", type=str, default="lr",
                        help="Which model do you want to use?")
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Where do you want to save the model?")
                        
    return parser.parse_args()