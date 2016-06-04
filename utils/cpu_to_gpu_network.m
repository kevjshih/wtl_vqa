function[net] = cpu_to_gpu_network(net)

for i = 1:length(net.layers)
    if isfield(net.layers{i}, 'weights')
        net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1});
        net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2});
    end
end