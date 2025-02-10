import torch
from torch import nn

def prune_magnitude(model, sparsity_ratio=0.5, device=torch.device("cuda:0"), prune_n=0, prune_m=0):

    subset = find_layers(model)

    for name in subset:
        W = subset[name].weight.data 
        W_metric = torch.abs(W)
        if prune_n != 0:
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            # numel: total elements in tensor
            # select the threshold value at the sparsity_ratio index
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
            W_mask = (W_metric<=thresh)

        W[W_mask] = 0


def prune_random(model, sparsity_ratio=0.5, device=torch.device("cuda:0"), prune_n=0, prune_m=0):

    subset = find_layers(model)

    for name in subset:
        W = subset[name].weight.data 
        W_metric = torch.abs(W)
        if prune_n != 0:
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            # numel: total elements in tensor
            # select the threshold value at the sparsity_ratio index
            W_mask = torch.cuda.FloatTensor(W_metric.shape[0], W_metric.shape[1]).uniform_() < sparsity_ratio

        W[W_mask] = 0



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        if name1 not in ['head']:
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
    return res



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # Save the current caching configuration and then disable caching for calibration.
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    # Load a calibration dataset ("c4") with a specified number of samples, sequence length, etc.
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    
    # Prepare calibration inputs, outputs, attention masks, and position IDs.
    # The shapes are commented for reference.
    # inps: (num_samples, sequence_length, hidden_dim)
    # outs: (num_samples, sequence_length, hidden_dim)
    # attention_mask: (1, 1, sequence_length, sequence_length)
    # position_ids: (1, sequence_length)
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # Get all transformer layers from the model.
    layers = model.model.layers
    for i in range(len(layers)):
        # Process each transformer layer one by one.
        layer = layers[i]
        
        # Extract the target submodules (e.g., specific linear layers) within the current transformer layer.
        subset = find_layers(layer)

        # In multi-GPU setups (e.g., for very large models like llama-30B/65B), move calibration data
        # to the device that the current layer is mapped to.
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        # Create a dictionary to hold WrappedGPT instances for each target submodule.
        # Each WrappedGPT instance collects activation statistics during forward passes.
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        # Define a hook function that will be called during the forward pass of a submodule.
        def add_batch(name):
            # The inner function "tmp" will be attached as a forward hook.
            def tmp(_, inp, out):
                # Use the first element of the input (assumed to be the relevant tensor) and the output,
                # and update the running statistics in the corresponding WrappedGPT instance.
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register the forward hooks for each target submodule.
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Run forward passes on the calibration data to collect activation statistics.
        for j in range(args.nsamples):
            with torch.no_grad():
                # Unsqueeze the input to add a batch dimension and process it through the current layer.
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                 position_ids=position_ids)[0]
        
        # Remove the forward hooks after the statistics have been collected.
        for h in handles:
            h.remove()
                
        # For each submodule in the current layer, compute a pruning metric and apply pruning.
        for name in subset:
            print(f"pruning layer {i} name {name}")
            # Compute a metric for pruning based on the absolute value of the weights
            # and the activation statistics (scaled by the square root of scaler_row).
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Initialize a mask with the same shape as W_metric where all entries are False.
            # This mask will mark weights for pruning (True means prune that weight).
            W_mask = (torch.zeros_like(W_metric) == 1)
                        
            if prune_n != 0:
                # Perform structured n:m pruning if prune_n is provided.
                # For each group (block) of prune_m weights, prune the prune_n weights with the smallest metric.
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        # Get a block of prune_m columns starting at the current index.
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        # Identify the indices of the prune_n smallest elements in the block.
                        indices_to_prune = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        # Update the mask to mark these indices as True (pruned).
                        W_mask.scatter_(1, ii + indices_to_prune, True)
            else:
                # For unstructured pruning, first sort the metric values.
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # Wanda variant: adjust the pruning threshold (alpha) using binary search
                    # to achieve a target sparsity ratio.
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    # Get the initial mask and current sparsity using the given alpha.
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    # Adjust alpha until the achieved sparsity is within a small tolerance of the target.
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # Simple unstructured pruning: prune a fixed fraction of weights by selecting
                    # the indices corresponding to the smallest metric values.
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            # Apply the mask: set the selected weights to zero (prune them).
            subset[name].weight.data[W_mask] = 0  
                
        # After pruning the current layer, run the forward pass again on the calibration data.
        # This updates the activations so they can serve as inputs for the next layer.
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                 position_ids=position_ids)[0]
        # Swap inps and outs: the output of this layer becomes the input for the next layer.
        inps, outs = outs, inps

    # Restore the original caching configuration.
    model.config.use_cache = use_cache 
    # Clear the CUDA cache to free up memory.
    torch.cuda.empty_cache()
