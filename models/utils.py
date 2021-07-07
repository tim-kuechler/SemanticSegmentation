import torch


def restore_checkpoint(optimizer, model, ckpt_dir, device='cuda'):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if optimizer is not None:
        optimizer.load_state_dict(loaded_state['optimizer'])
    model.load_state_dict(loaded_state['models'], strict=False)
    epoch = loaded_state['epoch']
    return epoch


def save_checkpoint(optimizer, model, epoch, ckpt_dir):
    saved_state = {
        'optimizer': optimizer.state_dict(),
        'models': model.state_dict(),
        'epoch': epoch
    }
    torch.save(saved_state, ckpt_dir)