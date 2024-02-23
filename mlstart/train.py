from args import parser
from model import Model
from data import SkData

def main():
    args = parser()
    
    # model and data
    model = Model(args.model_type)
    data = SkData(args.dataset_name)
    
    # fit and score
    model.fit(data=data)
    print("The model ", args.model_type, " has a score of ", model.score(data.X, data.y))

    if args.model_path:
        model.save(args.model_path)

if __name__ == "__main__":
    main()