import argparse

class BasicOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return opt    


class TrainOptions(BasicOptions):
    def initialize(self, parser):
        parser = BasicOptions.initialize(self, parser)
        parser.add_argument("--train_data_path", required=True)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--lr_decay", type=float, default=0.9)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--max_epoch", type=int, default=5)
        parser.add_argument("--env", type=str, default="default")
        parser.add_argument("--load_model_path", type=str, default="")
        parser.add_argument("--use_gpu", type=bool, default=False)
        parser.add_argument("--print_freq", type=float, default=20)
        parser.add_argument("--debug_file", type=str, default="tmp/debug")

        return parser

class TestOptions(BasicOptions):
    def initialize(self, parser):
        parser = BasicOptions.initialize(self, parser)
        parser.add_argument("--test_data_path", required=True)
        parser.add_argument("--load_model_path", required=True)
        parser.add_argument("--output_path", type=str, default="out/prediction.csv")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--use_gpu", type=bool, default=False)

        return parser
