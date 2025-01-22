import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Get training options
    opt = TrainOptions().parse()
    
    # Create dataset and calculate its size
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    
    # Create and setup model
    model = create_model(opt)
    model.setup(opt)
    
    # Create visualizer
    visualizer = Visualizer(opt)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize variables
    total_iters = 0
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()  # Update learning rates at the start of each epoch
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Move data to GPU (optimized)
            data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
            
            with torch.cuda.amp.autocast():  # Enable mixed precision
                if 'n_save_noisy' in vars(opt):
                    model.set_input(data, epoch)
                else:
                    model.set_input(data)
                model.optimize_parameters()  # Forward pass + Backward pass + Optimization
            
            # Display visuals
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            # Print losses and log
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            # Save the latest model
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()
        
        # Save model at the end of each epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        # Save visualizer losses
        visualizer.save_D_losses(model.get_current_losses())
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        # Clear cached memory
        torch.cuda.empty_cache()
