

class DefaultConfig(object):

    def __init__(self):
        self.gpu = -1                     # -1: use cpu
        self.epoch_num = 20
        self.batch_size = 128
        self.lr = 1e-3

        self.validate_batch_size = 32
        self.validate_batch_num = 8

        self.validate_step = 100
        self.save_step = 100

        self.early_stop_num = -1          # -1: don't use early stop
        self.use_hyperboard = False

        self.note = ''

    def todict(self,output_vars='all'):
        if output_vars=='all':
            return vars(self)
        else:
            all_var = vars(self)
            OutputVars = {}
            for key in output_vars:
                OutputVars[key] = all_var[key]
            return OutputVars

if __name__ == '__main__':
    c = DefaultConfig()
    print(c.todict())