from statistics import mean, stdev

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,patience = 7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            my_loss_list (list): contains the list of last 10 losses, to estimate the current mean and stddev of the validation loss
        """
        
        
        self.counter = 0
        self.early_stop = False
        self.patience = patience

    def stop(self, my_loss_list = []):

        losses = my_loss_list[0:len(my_loss_list)-1]
        val = my_loss_list[-1]
        avg = mean(losses)
        sigma = stdev(losses)
        if val < avg -abs(sigma) or val > avg + abs(sigma) :
            self.early_stop = False
        else :
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
