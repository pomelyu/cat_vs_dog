import warnings

class DefaultConfig(object):
    train_data_path = ""
    load_model_path = ""


    batch_size = 32
    lr = 0.0001
    lr_decay = 0.9

    max_epoch = 5
    use_gpu = False
    print_freq = 20
    env = "default"    

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
               warnings.warn("Warning: opt has not attribute {}".format(k))

            setattr(self, k, v)

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, ":", getattr(self, k)) 

opt = DefaultConfig()
