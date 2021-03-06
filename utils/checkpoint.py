import torch

def save(epoch, model, optim, loss, path):
    """
    Writes model checkpoint to given location
    
    Args:
        epoch: integer representing current training epoch
        model: pytorch model
        optim: pytorch optimizer
        loss: loss during epoch
        path: location to write checkpoint
    """

    print('saving checkpoint...')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
        }, path)

    print(f'checkpoint saved at {path}.')


def load(path):
    """
    Reads model checkpoint from given location

    Args:
        path: location to read checkpoint
    
    Returns:
        epoch: last epoch iteration
        model_state_dict: learned model parameters
        optimizer_state_dict: learned optimizer parameters
        loss: loss during epoch

    """

    print('loading checkpoint...')

    checkpoint = torch.load(path)
    
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print('checkpoint loaded.')

    return epoch, model_state_dict, optimizer_state_dict, loss

