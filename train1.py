import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    if opt.display_id > 0:  # Check if Visdom is enabled
        try:
            from visdom import Visdom
            vis = Visdom(port=opt.display_port)
            assert vis.check_connection()
        except Exception as e:
            print(f"Could not connect to Visdom server: {e}")
            opt.display_id = 0  # Disable Visdom visualization

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    scaler = torch.amp.GradScaler()  # Updated GradScaler usage

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            # Ensure tensors are moved to GPU
            if isinstance(data, dict):
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
            elif isinstance(data, list):
                data = [item.cuda(non_blocking=True) for item in data if isinstance(item, torch.Tensor)]

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, 0)

            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
