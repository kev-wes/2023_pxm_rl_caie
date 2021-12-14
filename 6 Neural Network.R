## Creats a neural network with length(sizes) layers. The first layer is an input layer and only distributes the inputs to the second layer.
## Hence, neither weights nor biases are computed for the first layer.
## Element [[4]] contains the activation functions for all Neurons.
net.create <- function(sizes, func = NULL){
  funcnames <- c("input", "sigmoid", "relu", "leakyrelu", "swish", "linear")
  if(is.null(func)){
    func <- rep("sigmoid", length(sizes))
    func[1] <- "input"
  }
  if(length(sizes) != length(func)){stop("Sizes and acitivation functions need to have same length")}
  if(is.character(func)){
    func <- tolower(func)
    if(!all(func%in% funcnames)){
      stop(paste0("All activation functions must be of the following: ", paste0(funcnames, collapse = ", "), "."))
    }
  }else{
    if(is.list(func)){
      for(i in 1:length(func)){
        func[[i]] <- tolower(func[[i]])
        if(!all(func[[i]]%in% funcnames)){
          stop(paste0("All activation functions must be of the following: ", paste0(funcnames, collapse = ", "), "."))
        }
      }
    }
  }
  net           <- list(sizes = sizes)
  net[[2]]      <- list() #weightgs
  net[[2]][[1]] <- matrix(rnorm(sizes[1]), ncol = 1)
  net[[3]]      <- list() #biases
  net[[4]]      <- list() #activations
  for(i in 1:length(sizes)){
    if(i == 1){
      net[[2]][[i]] <- matrix(rep(0, sizes[i]), nrow = sizes[i])
    }else{
      net[[2]][[i]] <- matrix(rnorm(sizes[i]*sizes[i-1], sd = 1/sqrt(sizes[i-1])), nrow = sizes[i])
    }
    net[[3]][[i]] <- matrix(rnorm(sizes[i]), nrow = sizes[i])*(i!=1)
  }
  
  if(is.null(func)){
    for(i in 1:length(sizes)){
      net[[4]][[i]] <- rep("sigmoid", sizes[i])
    }
  }else{
    if(class(func)=="list"){
      net[[4]] <- func
    }else{
      for(i in 1:length(func)){
        net[[4]][[i]] <- rep(func[i], sizes[i])
      }
    }
  }
  names(net) <- c("sizes", "weights", "biases", "activations")
  return(net)
}


#### Computes the output from a neural network
#### Inputs are a neural network (net) and a single data point (data), i.e. in the case of digits a single image
#### If "full" is set to T, all intermediate output is returned, as well. This is useful to plot the whole network
net.IO <- function(net, data, full = F){
  biases <- rep(0, net$sizes[1])
  output <- rep(0, net$sizes[1])
  if(!is.null(dim(data))){ ##data ist a matrix with multiple observations in rows
    if(full){out <- list()}else{out <- numeric(0)}
    for(row in 1:dim(data)[1]){
      input  <- data[row,]
      if(!full){
        for(i in 2:length(net$sizes)){
          output <- rep(0, net$sizes[i])
          for(j in 1:net$sizes[i]){
            parameters <- list(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,])
            output[j]  <- net.activation(parameters = parameters, data = input, func = net$activations[[i]][j])
          }
          input <- output
        }
        out <- c(out, output)
      }else{
        outputs <- list(input)
        for(i in 2:length(net$sizes)){
          output <- rep(0, net$sizes[i])
          for(j in 1:net$sizes[i]){
            parameters <- list(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,])
            output[j]  <- net.activation(parameters = parameters, data = input, func = net$activations[[i]][j])
          }
          outputs[[i]] <- output
          input        <- output
        }
        out[[row]] <- outputs
      }
    }
    return(out)
  }else{
    input  <- data
    if(!full){
      for(i in 2:length(net$sizes)){
        output <- rep(0, net$sizes[i])
        for(j in 1:net$sizes[i]){
          parameters <- list(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,])
          output[j]  <- net.activation(parameters = parameters, data = input, func = net$activations[[i]][j])
        }
        input <- output
      }
      return(output)
    }else{
      outputs <- list(input)
      for(i in 2:length(net$sizes)){
        output <- rep(0, net$sizes[i])
        for(j in 1:net$sizes[i]){
          parameters <- list(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,])
          output[j]  <- net.activation(parameters = parameters, data = input, func = net$activations[[i]][j])
        }
        outputs[[i]] <- output
        input        <- output
      }
      return(outputs)
    }
  }
}


net.activation <- function(parameters, data, func){
  if(func == "sigmoid"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.sigmoid(weights, biases, data))
  }
  if(func == "linear"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(weights%*%data+biases)
  }
  if(func == "swish"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.swish(weights, biases, data))
  }
  if(func == "relu"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.relu(weights, biases, data))
  }
  if(func == "leakyrelu"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.leakyrelu(weights, biases, data))
    
  }
}



net.activation.x <- function(x, func){
  if(func == "sigmoid"){
    return(1/(1+exp(-x)))
  }
  if(func == "linear"){
    return(x)
  }
  if(func == "swish"){
    return(x/(1+exp(-x)))
  }
  if(func == "relu"){
    return(ifelse(x>0, x, 0))
  }
  if(func == "leakyrelu"){
    return(ifelse(x>0, x, x*0.01))
  }
}



net.activation.deriv <- function(parameters, data, func){
  if(func == "sigmoid"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.sigmoid.deriv(weights, biases, data))
  }
  if(func == "linear"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(weights%*%(data*0) + 1)
  }
  if(func == "swish"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.swish.deriv(weights, biases, data))
  }
  if(func == "relu"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.relu.deriv(weights, biases, data))
  }
  if(func == "leakyrelu"){
    weights <- parameters[[1]]
    biases  <- parameters[[2]]
    return(net.leakyrelu.deriv(weights, biases, data))
  }
}




net.activation.deriv.x <- function(x, func){
  if(func == "sigmoid"){
    net.sigmoid.eval <- net.sigmoid.x(x)
    return(net.sigmoid.eval*(1-net.sigmoid.eval))
  }
  if(func == "linear"){
    return(rep(1, length(x)))
  }
  if(func == "swish"){
    net.sigmoid.eval <- net.sigmoid.x(x)
    return(net.sigmoid.eval * (1 + x * (1 - net.sigmoid.eval)))
  }
  if(func == "relu"){
    return(ifelse(x > 0, 1, 0))
  }
  if(func == "leakyrelu"){
    return(ifelse(x > 0, 1, 0.01))
  }
}

## Compute weights*data + biases ##
to.x <- function(weights, biases, data){
  return(weights %*% data + biases)
}

#### The activation functions. ####
### Inputs are the weights and biases of the current neuron or layer as well as all inputs (data)
### Weights is a matrix of dimension N_neurons x N_inputs
### Biases is a matrix of dimension N_neurons x 1
### Data is a vector of length(N_inputs)
### or a vector()
net.sigmoid <- function(weights, biases, data){
  return(1/(1+exp(-(weights%*%data+biases))))
}

net.sigmoid.x <- function(x){
  return(1/(1+exp(-x)))
}

net.swish <- function(weights, biases, data){
  x = to.x(weights, biases, data)
  return(x * net.sigmoid.x(x))
}

net.relu <- function(weights, biases, data){
  x = to.x(weights, biases, data)
  return(ifelse(x  > 0, x, 0))
}

net.leakyrelu <- function(weights, biases, data){
  x = to.x(weights, biases, data)
  return(ifelse(x > 0, x, 0.01 * x))
}


#### Derivative of the input functions  wrt. z = t(weights)%*%data + biases. ####
net.sigmoid.deriv <- function(weights, biases, data){
  net.sigmoid.eval <- net.sigmoid(weights, biases, data)
  return(net.sigmoid.eval*(1-net.sigmoid.eval))
}

net.swish.deriv <- function(weights, biases, data){
  x    = to.x(weights, biases, data)
  sigx = net.sigmoid.x(x)
  return(sigx * (1 + x * (1 - sigx)))
}

net.relu.deriv <- function(weights, biases, data){
  x = to.x(weights, biases, data)
  return(ifelse(x  > 0, 1, 0))
}

net.leakyrelu.deriv <- function(wieghts, biases, data){
  x = to.x(weights, biases, data)
  return(ifelse(x > 0, 1, 0.01 ))
}




### The cost function of the neural network. Distinguish between squares, ie c = ||data-net.output|| and crossentropy
### crossentropy increases learning speed in early epochs.
### Inputs are a data vector, a forecast vector (net.output) and the cost function that's used (cf; default is "squares")
net.cost    <- function(data, net.output, cf = "squares"){
  if(cf == "squares"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(sum((data-net.output)^2))
  }
  if(cf == "ce"){ #cross entropy
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(-sum(data*log(net.output)+(1-data)*log(1-net.output)))
  }
  if(cf == "none"){ #for use in reinforcement learning -> maximize the output
    return(sum(net.output))
  }
}


### Derivative of the cost function wrt. net.output
### Inputs are a data vector, a forecast vector (net.output) and the cost function that's used (cf; default is "squares")
net.cost.prime <- function(data, net.output, cf = "squares"){
  if(cf == "squares"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(2*(net.output-data))
  }
  if(cf == "ce"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(-(data / net.output - (1 - data) / (1 - net.output)))
  }
  if(cf == "none"){### For use in reinforcement learning -> maximize the output
    return(1)
  }
}


### Compute the gradient of the cost function wrt w and b
### Inputs are the net, a list of data, containing a matrix of image inputs and a vector of digits and a char to determine the cost function
net.gradient <- function(net, data, cf = "squares"){
  net.size         <- length(net$sizes)
  grad_w           <- net$weights
  grad_b           <- net$biases
  activations      <- list()
  input            <- data[[1]]
  trueval          <- data[[2]]
  activations[[1]] <- input
  activations[[2]] <- input
  
  for(i in 2:net.size){
    parameters <- list(weights = net$weights[[i]], biases = net$biases[[i]])
    activations[[i+1]] <- net.activation(parameters = parameters, data = activations[[i]], func = net$activations[[i]][1])
  }
  if(cf == "ce"){
    activations[[net.size+1]] <- pmax(pmin(activations[[net.size+1]],0.9999999), 0.0000001)
  }
  net.output             <- activations[[net.size+1]]
  delta                  <- net.cost.prime(data = trueval, net.output = activations[[net.size+1]], cf = cf) *
    net.activation.deriv(parameters = parameters, data = activations[[i]], func = net$activations[[i]][1])
  grad_b[[net.size]][,1] <- delta
  grad_w[[net.size]]     <- delta%*%t(activations[[net.size]])
  if(net.size <3){return(list(grad_b = grad_b,grad_w = grad_w))}
  for(i in (net.size-1):2){
    parameters  <- list(weights = net$weights[[i]], biases = net$biases[[i]])
    output      <- net.activation.deriv(parameters = parameters, data = activations[[i]], func = net$activations[[i]][1])
    delta       <- (t(net$weights[[i+1]])%*%delta) * output
    grad_b[[i]] <- delta
    grad_w[[i]] <- delta%*%t(activations[[i]])
  }
  grad_b[[1]] <- grad_b[[1]] * 0
  grad_w[[1]] <- grad_w[[1]] * 0
  return(list(grad_b = grad_b,grad_w = grad_w))
}

net.gradient.RL <- function(net, data){
  datlist <- list(data, NA)
  return(net.gradient(net = net, data = datlist, cf = "none"))
}

net.add.gradient <- function(net, data, cf, stepsize){
  grad <- net.gradient(net, data, cf)
  for(i in 1:length(net$sizes)){
    net$weights[[i]] <- net$weights[[i]] + grad$grad_w[[i]] * stepsize
    net$biases[[i]]  <- net$biases[[i]]  + grad$grad_b[[i]] * stepsize
  }
  return(net)
}

net.add.gradient.RL <- function(net, grad, stepsize){
  # grad <- net.gradient.RL(net, data)
  for(i in 1:length(net$sizes)){
    net$weights[[i]] <- net$weights[[i]] + grad$grad_w[[i]] * stepsize
    net$biases[[i]]  <- net$biases[[i]]  + grad$grad_b[[i]] * stepsize
  }
  return(net)
}

net.add.trace <- function(net, elig.trace, stepsize){
  for(i in 1:length(net$sizes)){
    net$weights[[i]] <- net$weights[[i]] + elig.trace$grad_w[[i]] * stepsize
    net$biases[[i]]  <- net$biases[[i]]  + elig.trace$grad_b[[i]] * stepsize
  }
  return(net)
}

net.update.trace1 <- function(elig.trace, gradient, gamma0, type="accumulating"){
  if(type == "accumulating"){
    for(i in 1:length(elig.trace$grad_w)){
      elig.trace$grad_w[[i]] <- elig.trace$grad_w[[i]] * gamma0 + gradient$grad_w[[i]]
      elig.trace$grad_b[[i]] <- elig.trace$grad_b[[i]] * gamma0 + gradient$grad_b[[i]]
    }
  }else{
    stop("types, other than accumulating are not implemented, yet!")
  }
  return(elig.trace)
}

net.update.trace2 <- function(elig.trace, net, data, gamma0, type="accumulating", cf = "none"){
  gradient <- net.gradient(net, data, cf)
  return(net.update.trace1(elig.trace = elig.trace, gradient = gradient, gamma0 = gamma0, type = type))
}

net.initialize.trace <- function(net){
  dta        <- list(rep(0, net.sizes[1]), 0)
  elig.trace <- net.gradient(net, dta, "none")
  for(i in 1:length(net.sizes)){
    elig.trace$grad_w[[i]] <- elig.trace$grad_w[[i]] * 0
    elig.trace$grad_b[[i]] <- elig.trace$grad_b[[i]] * 0
  }
  return(elig.trace)
}



net.save <- function(ANN, path, NetName){
  require(tidyverse)
  ANNLayers      <- as_data_frame(t(ANN$sizes))
  ANNSize        <- length(ANNLayers)
  ANNWeights     <- ANN$weights
  ANNBiases      <- ANN$biases
  ANNActivations <- ANN$activations
  activ <- rep("", ANNSize)
  for(i in 1:ANNSize){
    activ[i] <- ANNActivations[[i]][1]
  }
  ANNActivations <- as_data_frame(t(activ))
  #check if directories exist
  if(!dir.exists(paste0(path, "/", NetName ))){
    dir.create(paste0(path, "/", NetName), recursive = TRUE)
  }
  if(!dir.exists(paste0(path, "/", NetName, "/weights"))){
    dir.create(paste0(path, "/", NetName, "/weights"))
  }
  if(!dir.exists(paste0(path, "/", NetName, "/biases"))){
    dir.create(paste0(path, "/", NetName, "/biases"))
  }
  #check if files exist and if they do return an error
  if(!file.exists(paste0(path, "/", NetName, "/Layers.csv"))){
    write_csv(ANNLayers, path = paste0(path, "/", NetName, "/Layers.csv"))
  }else{
    return("Layers.csv already exists! Choose different path or net name!")
  }
  if(!file.exists(paste0(path, "/", NetName, "/Activations.csv"))){
    write_csv(ANNActivations, path = paste0(path, "/", NetName, "/Activations.csv"))
  }else{
    return("Activations.csv already exists! Choose different path or net name!")
  }
  for(i in 1:ANNSize){
    ThisBiases  <- as_data_frame(ANNBiases[[i]])
    ThisWeights <- as_data_frame(ANNWeights[[i]])
    write_csv(ThisBiases, path = paste0(path, "/", NetName, "/biases/",i,".csv") )
    write_csv(ThisWeights, path = paste0(path, "/", NetName, "/weights/",i,".csv") )
  }
  return("All nets succesfully stored!")
}


net.load <- function(path){
  require(tidyverse)
  if( !dir.exists(path)){
    stop("Path doesn't exist!")
  }
  ANNLayers      <- unlist(read_csv(paste0(path, "/Layers.csv")))
  ANNActivations <- unlist(read_csv(paste0(path, "/Activations.csv")))
  ANNSize        <- length(ANNLayers)
  ThisNet        <- net.create(ANNLayers, ANNActivations)
  for (i in 1:ANNSize){
    ThisNet$weights[[i]] <- as.matrix(read_csv(paste0(path, "/weights/", i, ".csv")))
    ThisNet$biases[[i]]  <- as.matrix(read_csv(paste0(path, "/biases/", i, ".csv")))
  }
  return(ThisNet)
}