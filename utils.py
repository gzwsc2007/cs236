import torch
import plotly.graph_objects as go
import plotly.express as px
import os

def plot_np_multi(xy_dict, xlabel='time (s)'):
    fig = go.Figure()
    for name, (x, y) in xy_dict.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=name))
    fig.update_layout(xaxis_title=xlabel)
    fig.show()

def save_model_by_name(model, epoch, isBest=False):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(epoch))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

    if isBest:
        print("Best model so far: ", file_path)
        with open(os.path.join(save_dir, 'best_model.txt'), 'w+') as f:
            f.write(file_path)

def load_model_by_name(model, epoch, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        epoch: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(epoch))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def save_states(name, epoch, model, optimizer):
    save_dir = os.path.join('checkpoints', name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(epoch))
    torch.save({'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict()}, file_path)
    print('Model saved to ', file_path)