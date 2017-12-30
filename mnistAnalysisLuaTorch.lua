require 'torch'
require 'nn'
require 'optim'
mnist = require 'mnist'
require 'sys'
--require 'cutorch'

start_time = os.time()  -- clock time start

fullset = mnist.traindataset()  -- training and validation set
testset = mnist.testdataset()   --testing set

-- establish training data
trainset = {
    size = 50000,
    data = fullset.data[{{1,50000}}]:double(),
    label = fullset.label[{{1,50000}}]
}

--establish validation set
validationset = {
    size = 10000,
    data = fullset.data[{{50001,60000}}]:double(),
    label = fullset.label[{{50001,60000}}]
}


-- MLP model
model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 100))
model:add(nn.Tanh())
model:add(nn.Linear(100, 10))
model:add(nn.LogSoftMax())
--model:cuda()

criterion = nn.ClassNLLCriterion()
--criterion:cuda()

batch = 32


--set training paramters
training_params = {
   learningRate = 1e-1,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 9e-1
}

x, dl_dx = model:getParameters()


--confusion matrix
require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10)
-- create a function to compute 
function classEval(model, inputs, targets)
   cm:zero()
   for i=1,inputs:size(1) do
      local input, target = inputs[i], targets:narrow(1,i,1)
      local output = model:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end
print(cm) -- returns empty values..

--training the data
step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or batch
    
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        --local size = math.min(1, trainset.size) --use for stochastic gradient
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        --inputs = inputs:cuda()
        local targets = torch.Tensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, training_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the training optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

--evaluate the data
eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or batch
    
    for i = 1,dataset.size,batch_size do
        --local size = math.min(1, dataset.size) -- use for pure stochastic gradient
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

--number of epochs
max_iters = 30

do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

testset.data = testset.data:double()

print('Accuracy on Test Set: ' .. eval(testset))

--get end time and run time
end_time = os.time()
elapsed_time = os.difftime(end_time-start_time)
elapsed_time = elapsed_time/60
print('Elapsed time (minutes) : ' .. elapsed_time)