import numpy as np
import torch
import model_helper_functions as mf


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to. If None, saving is skipped.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.acomp_metrics = None
        self.best_acomp_metrics = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer=None, train_losses=None, val_losses=None, train_clf_reports=None, val_clf_reports=None, acomp_metrics=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.acomp_metrics = acomp_metrics
            self.best_acomp_metrics = acomp_metrics
            self.save_checkpoint(val_loss, model, optimizer, train_losses, val_losses, train_clf_reports, val_clf_reports)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(
                    f'Early stopping with best value: {abs(self.best_score)} and acompanying metrics: {self.acomp_metrics}')
        else:
            self.best_score = score
            self.acomp_metrics = acomp_metrics
            self.save_checkpoint(val_loss, model, optimizer, train_losses, val_losses, train_clf_reports, val_clf_reports)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, train_losses, val_losses, train_clf_reports, val_clf_reports):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        best_acomp_notif = ''
        if self.path:
            # torch.save(model.state_dict(), self.path)
            if self.best_acomp_metrics[1] < self.acomp_metrics[1]:
                self.best_acomp_metrics = self.acomp_metrics
                best_acomp_notif = f'best_{self.best_acomp_metrics[0]}_'
                
            mf.save_checkpoint(self.path, model, optimizer, val_loss, bam=best_acomp_notif)
            mf.save_metrics(self.path, train_losses, val_losses, train_clf_reports, val_clf_reports, bam=best_acomp_notif)

        self.val_loss_min = val_loss
