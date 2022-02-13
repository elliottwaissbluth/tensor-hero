from pathlib import Path
import sys
sys.path.insert(1, str(Path.cwd()))
from tensor_hero.model import ColabLazyDataset, Transformer, LazierDataset
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json
import pickle

def __load_model(model_directory):
    '''
    Loads model and param dict from model_directory. useful for continuing training
    Helper function for initialize_params()
    
    ~~~~ ARGUMENTS ~~~~
    - model_directory (Path): Folder containing model weights and params
        - probably ./model/saved_models/<model name>
    
    ~~~~ RETURNS ~~~~
    - dict: params loaded from model directory
    '''
    with open(str(model_directory / 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    f.close()

    return params

def initialize_params(params):
    '''
    Takes the original params and modifies them to match the current training objective.
        - Will load params from model directory if desired
        - Initializes new params for training a new model
        - Asks for user input regarding descriptions of new models

    ~~~~ ARGUMENTS ~~~~
    - params (dict): Dictionary containing relevant model information. See definition in main() for more information.

    ~~~~ RAISES ~~~~
    - SystemExit: If the experiment is to be aborted.

    ~~~~ RETURNS ~~~~
    - dict: model and training parameters
    '''
    # Ask user whether they will load a pretrained model to continue training or initialize a new model
    response = str(input('Load from pretrained model? (y/n): ')).lower()
    while response not in ['y', 'n']:
        response = str(input('invalid input\nAre the parameters correct (y/n)?: ')).lower()
        
    if response == 'y':  # If loading from pretrained model, load its weights from its directory
        response = str(input('Enter name of model to load: '))  # This will be the name of the directory under ./model/saved_models
        while not os.path.isdir(Path.cwd() / 'model' / 'saved_models' / response):
            print('Error: {} is not a valid directory'.format(response))
            response = str(input('Enter name of model to load: '))
        model_directory = Path.cwd() / 'model' / 'saved_models' / response
        params = __load_model(model_directory)
        params['LOAD'] = True

    else:  # If initializing new model, create a new directory for it
        while os.path.isdir(Path.cwd() / 'model' / 'saved_models' / params['model_name']):
            new_name = input('Directory already exists.\nEnter new model name: ')
            params['model_name'] = str(new_name)
        os.mkdir(Path.cwd() / 'model' / 'saved_models' / params['model_name'])
        params['LOAD'] = False

    params['model_file_name'] = params['model_name'] + '.pt'  # Holds weights of model
    params['model_outfile'] = str(Path.cwd() / 'model' / 'saved_models' / params['model_name'] / params['model_file_name'])  # model directory

    # Validate parameters
    print(json.dumps(params, indent=4))
    response = str(input('Are the parameters correct (y/n)?: ')).lower()
    while response not in ['y', 'n']:
        response = str(input('invalid input\nAre the parameters correct (y/n)?: ')).lower()
    if response == 'n':
        raise SystemExit(0)

    # Gather description of experiment from user
    if not 'experiment_description' in params.keys():
        experiment_description = input('Enter experiment description: ')
        params['experiment_description'] = experiment_description

    # Save parameters
    with open(str(Path.cwd() / 'model'/ 'saved_models' / params['model_name'] / 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    f.close()
    
    print('parameters saved\n')
    
    return params

def initialize_model(params, device):
    '''
    Takes params and the device (CUDA or CPU) and initializes a transformer model, as defined in ./tensor_hero/model.py
    
    ~~~~ ARGUMENTS ~~~~
    - params (dict): Model and training parameters. Should be the output from initialize_params()
    - device (str): "CUDA" or "CPU"
    
    ~~~~ RETURNS ~~~~
        PyTorch model: Transformer model initialized with params and sent to device. Defined in ./tensor_hero/model.py
    '''
    model = Transformer(
            embedding_size = params['embedding_size'],
            trg_vocab_size = params['trg_vocab_size'],
            num_heads = params['num_heads'],
            num_encoder_layers = params['num_encoder_layers'],
            num_decoder_layers = params['num_decoder_layers'],
            forward_expansion = params['embedding_size']*params['forward_expansion'],
            dropout = params['dropout'],
            max_len = params['max_src_len'],
            device = device,
        ).to(device)

    if params['LOAD']:
        state_dict_path = str(Path.cwd() / 'model'/ 'saved_models' / params['model_name'] / ((params['model_name']) + '.pt'))
        model.load_state_dict(torch.load(state_dict_path))
    
    return model

def main():
        # NOTE: Without if __name__ == '__main__', multithreading makes this script impossible to run
        
        # ---------------------------------------------------------------------------- #
        #                              INITIAL PARAMETERS                              #
        # ---------------------------------------------------------------------------- #
        
        params = {
            'training_data' : 'train separated',     # CHANGEME (these parameters must be changed each experiment)
            'model_name' : 'model12',                # CHANGEME
            'optimizer' : 'Adam',                    # CHANGEME (maybe not this one, but you do have to fill it in manually)
            'train_path' : r'X:\Training Data\training_ready\train',

            'num_epochs' : 500,
            'batch_size' : 12,
            'shuffle' : True,
            'num_workers' : 4,
            'drop_last' : True,
            'last_global_step' : 0,

            'max_trg_len' : 500, # NOTE: max_trg_len <= max_src_len otherwise side asset error is triggered
            'max_src_len' : 500,
            'trg_vocab_size' : 435,
            'pad_idx' : 434,
            'embedding_size' : 512,

            'lr' : 1e-4,
            'num_heads' : 8,
            'num_encoder_layers' : 2,
            'num_decoder_layers' : 2,
            'dropout' : 0.1,
            'forward_expansion' : 4,

            'date' : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }
        
        params = initialize_params(params)
        
        # ---------------------------------------------------------------------------- #
        #                            TENSORBOARD PARAMETERS                            #
        # ---------------------------------------------------------------------------- #
        
        writer = SummaryWriter(r'runs/'+params['model_name'])
        hparam_dict = {
            'batch_size' : params['batch_size'],
            'embedding_size' : params['embedding_size'],
            'learning_rate' : params['lr'],
            'num_heads' : params['num_heads'],
            'num_encoder_layers' : params['num_encoder_layers'],
            'dropout' : params['dropout'],
            'forward_expansion' : params['embedding_size']*params['forward_expansion']
        }

        
        # ---------------------------------------------------------------------------- #
        #                                  DATALOADER                                  #
        # ---------------------------------------------------------------------------- #
        
        dl_params = {
            'batch_size' : params['batch_size'],
            'shuffle' : params['shuffle'],
            'num_workers' : params['num_workers'],
            'drop_last' : params['drop_last'],
        }

        # Define data loaders
        train_data = ColabLazyDataset(Path(params['train_path']), params['max_src_len'], params['max_trg_len'], params['pad_idx'])
        train_loader = torch.utils.data.DataLoader(train_data, **dl_params)

        # ---------------------------------------------------------------------------- #
        #                              TRAINING PARAMETERS                             #
        # ---------------------------------------------------------------------------- #
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training hyperparameters
        learning_rate = params['lr']
        num_epochs = params['num_epochs']

        model = initialize_model(params, device) 

        # torch.save(model.state_dict(), 'model.pt')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # criterion = nn.CrossEntropyLoss() # Multi-class loss, when you have a many class prediction problem
        criterion = nn.CrossEntropyLoss(ignore_index=params['pad_idx'])

        # ---------------------------------------------------------------------------- #
        #                                 TRAINING LOOP                                #
        # ---------------------------------------------------------------------------- #
        
        model.train() # Put model in training mode, so that it knows it's parameters should be updated
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Batches come through as a tuple defined in the return statement __getitem__ in the Dataset
                spec, notes = batch[0].to(device), batch[1].to(device)

                # forward prop
                output = model(spec, notes[..., :-1])           # Don't pass the last element into the decoder, want it to be predicted
                # output = output.reshape(-1, output.shape[2])  # Reshape the output for use by criterion
                notes = notes[..., 1:] # .reshape(-1)           # Same for the notes
                optimizer.zero_grad()                           # Zero out the gradient so it doesn't accumulate

                loss = criterion(output.permute(0,2,1), notes)  # Calculate loss, this is output vs ground truth
                
                if batch_idx%25 == 0:
                    print('\nEpoch {}, Batch {}'.format(epoch+1, batch_idx))
                    print('Training Loss: {}'.format(loss.item()))
                
                loss.backward()     # Compute loss for every node in the computation graph

                # This line to avoid the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()    # Update model parameters
                params['last_global_step'] += 1

                # Write to tensorboard
                writer.add_scalar("Training Loss", loss, global_step=params['last_global_step'])
                # writer.add_hparams(hparam_dict=hparam_dict, metric_dict={'training_loss' : loss}) # NOTE: Move this outside the for loop

                # if batch_idx%100 == 0:
                    # print('Ground Truth (sample) : {}'.format(notes[0]))
                    # print('Canidate (sample)     : {}'.format(torch.argmax(output[0], dim=1)))
                
            # Print training update
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {loss.item()}')
            
            # Save the model every epoch
            torch.save(model.state_dict(), str(params['model_outfile']))
            with open(str(Path.cwd() / 'model'/ 'saved_models' / params['model_name'] / 'params.pkl'), 'wb') as f:
                pickle.dump(params, f)
            print('model saved')
            
            # # Evaluate on validation set
            # model.eval()
            # for batch_idx, batch in enumerate(val_loader):
                # spec, notes = batch[0].to(device), batch[1].to(device)

                # # forward prop
                # output = model(spec, notes[..., :-1]) # Don't pass the last element into the decoder, want it to be predicted

                # # output = output.reshape(-1, output.shape[2]) # Reshape the output for use by criterion
                # notes = notes[..., 1:] # .reshape(-1)           # Same for the notes
                
                # loss = criterion(output.permute(0,2,1), notes)     # Calculate loss, this is output vs ground truth

                # writer.add_scalar("Validation Loss", loss, global_step=step)
                # step += 1
            # print('Validation Loss: {}'.format(loss.item()))

if __name__ == '__main__':
    main()