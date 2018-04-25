require(sigmoid)

mapVectorToNueral <- function(wieghts) {
  s = 1
  e = inputSize * hiddenSize
  map = list()
  map$hiddenWieghts <- matrix(wieghts[s:e], 
                              ncol = hiddenSize, 
                              byrow = TRUE)
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  
  map$recurrentWieghts <- matrix(wieghts[s:e], 
                                 ncol = hiddenSize,
                                 byrow = TRUE)
  
  s = e + 1
  e = e + outSize * hiddenSize
  map$outputWieghts <- matrix(wieghts[s:e], 
                              ncol = outSize, 
                              byrow = TRUE)
  return(map)
}



rnnClassify <- function(hiddenWieghts, recurrentWieghts, outputWieghts, recurrent){
  result <- c()
  
  for(i in 1:length(dataset[,1])){
    
    sample = as.double(dataset[i,])
    
    inputResult <- sample %*% hiddenWieghts + recurrent %*% recurrentWieghts
    
    inputResult <- sigmoid(inputResult)
    
    recurrent <- inputResult
    
    output <- inputResult %*% outputWieghts
    output <- ifelse(output < 0.5, 0, 1)
    
    output <-  sum(2^(0:(outSize-1)) * output)
    result <- c(result, output[1])
  }
  return(result)
}

classify <- function(wieghts){
  map = mapVectorToNueral(wieghts)
  result <- rnnClassify(map$hiddenWieghts, map$recurrentWieghts, map$outputWieghts, rep(0, hiddenSize))
  
  return(result)
}

